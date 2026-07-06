# -*- coding: utf-8 -*-
"""
ABSA Web 服务 — 第一周项目
Flask 后端：接收句子+方面词+模型类型，返回情感分析结果
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, pad_and_truncate
from models import LSTM
from models.bert_spc import BERT_SPC
from transformers import BertModel

# ============================================================
# Flask 初始化
# ============================================================
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # 模板更新后自动重新加载
CORS(app)

# ============================================================
# 全局变量：模型和配置
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABELS = ['Negative', 'Neutral', 'Positive']

DATASET_FILES = {
    'laptop': {
        'train': './datasets/semeval14/Laptops_Train.xml.seg',
        'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
    }
}

# 模型实例（启动时加载）
lstm_model = None
lstm_tokenizer = None
bert_model = None
bert_tokenizer = None
bert_base = None

# ============================================================
# 模型加载（应用启动时执行一次）
# ============================================================
def load_lstm_model():
    """加载 LSTM + GloVe 模型"""
    global lstm_model, lstm_tokenizer

    opt = type('', (), {})()
    opt.model_name = 'lstm'
    opt.dataset = 'laptop'
    opt.dataset_file = DATASET_FILES['laptop']
    opt.inputs_cols = ['text_indices']
    opt.max_seq_len = 85
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.bert_dim = 768
    opt.pretrained_bert_name = 'bert-base-uncased'
    opt.polarities_dim = 3
    opt.hops = 3
    opt.device = DEVICE
    opt.local_context_focus = 'cdm'
    opt.SRD = 3

    # 构建/加载 tokenizer 和词向量
    lstm_tokenizer = build_tokenizer(
        fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
        max_seq_len=opt.max_seq_len,
        dat_fname='{}_tokenizer.dat'.format(opt.dataset))
    embedding_matrix = build_embedding_matrix(
        word2idx=lstm_tokenizer.word2idx,
        embed_dim=opt.embed_dim,
        dat_fname='{}_{}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))

    # 加载模型权重
    lstm_model = LSTM(embedding_matrix, opt).to(DEVICE)
    lstm_model.load_state_dict(torch.load('state_dict/lstm_laptop_val_acc_0.685', map_location=DEVICE))
    lstm_model.eval()

    return opt

def load_bert_model():
    """加载 BERT_SPC 模型"""
    global bert_model, bert_tokenizer, bert_base

    opt = type('', (), {})()
    opt.model_name = 'bert_spc'
    opt.dataset = 'laptop'
    opt.dataset_file = DATASET_FILES['laptop']
    opt.inputs_cols = ['concat_bert_indices', 'concat_segments_indices']
    opt.max_seq_len = 85
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.bert_dim = 768
    opt.pretrained_bert_name = 'bert-base-uncased'
    opt.polarities_dim = 3
    opt.hops = 3
    opt.dropout = 0.1
    opt.device = DEVICE
    opt.local_context_focus = 'cdm'
    opt.SRD = 3

    bert_tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    bert_base = BertModel.from_pretrained(opt.pretrained_bert_name)
    bert_model = BERT_SPC(bert_base, opt).to(DEVICE)
    bert_model.load_state_dict(torch.load('state_dict/bert_spc_laptop_val_acc_0.7868', map_location=DEVICE))
    bert_model.eval()

    return opt

# ============================================================
# 推理函数
# ============================================================
def predict_lstm(text, aspect):
    """使用 LSTM 模型进行情感预测"""
    text_lower = text.lower().strip()
    aspect_lower = aspect.lower().strip()

    # 按方面词切分句子
    text_left, _, text_right = [s.strip() for s in text_lower.partition(aspect_lower)]
    full_text = text_left + ' ' + aspect_lower + ' ' + text_right
    text_indices = lstm_tokenizer.text_to_sequence(full_text)

    t_inputs = [torch.tensor([text_indices], device=DEVICE) for _ in ['text_indices']]
    with torch.no_grad():
        outputs = lstm_model(t_inputs)
        probs = F.softmax(outputs, dim=-1).cpu().numpy()[0]

    return {
        'sentiment': LABELS[probs.argmax()],
        'probabilities': {
            'Negative': round(float(probs[0]), 4),
            'Neutral': round(float(probs[1]), 4),
            'Positive': round(float(probs[2]), 4)
        }
    }

def predict_bert(text, aspect):
    """使用 BERT_SPC 模型进行情感预测"""
    text_lower = text.lower().strip()
    aspect_lower = aspect.lower().strip()

    text_left, _, text_right = [s.strip() for s in text_lower.partition(aspect_lower)]

    # 构造 BERT 输入：[CLS] 句子 [SEP] 方面词 [SEP]
    concat_text = '[CLS] ' + text_left + ' ' + aspect_lower + ' ' + text_right + ' [SEP] ' + aspect_lower + ' [SEP]'
    concat_bert_indices = bert_tokenizer.text_to_sequence(concat_text)

    text_len = np.sum(bert_tokenizer.text_to_sequence(
        '[CLS] ' + text_left + ' ' + aspect_lower + ' ' + text_right + ' [SEP]') != 0)
    aspect_len = np.sum(bert_tokenizer.text_to_sequence(aspect_lower) != 0)
    concat_segments_indices = pad_and_truncate(
        [0] * (text_len + 2) + [1] * (aspect_len + 1),
        bert_tokenizer.max_seq_len)

    t_inputs = [
        torch.tensor([concat_bert_indices], device=DEVICE),
        torch.tensor([concat_segments_indices], device=DEVICE)
    ]
    with torch.no_grad():
        outputs = bert_model(t_inputs)
        probs = F.softmax(outputs, dim=-1).cpu().numpy()[0]

    return {
        'sentiment': LABELS[probs.argmax()],
        'probabilities': {
            'Negative': round(float(probs[0]), 4),
            'Neutral': round(float(probs[1]), 4),
            'Positive': round(float(probs[2]), 4)
        }
    }

# ============================================================
# API 路由
# ============================================================
@app.route('/')
def index():
    """返回前端页面"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """返回可用模型列表"""
    return jsonify({
        'models': [
            {'id': 'lstm', 'name': 'LSTM + GloVe', 'description': '轻量级 LSTM 编码器，使用 GloVe 词向量'},
            {'id': 'bert_spc', 'name': 'BERT_SPC', 'description': 'BERT 预训练模型，上下文感知能力更强'}
        ]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    情感分析接口
    请求体: {"sentence": "...", "aspect": "...", "model_type": "lstm"|"bert_spc"}
    返回: {"sentiment": "Positive", "probabilities": {...}}
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求体不能为空'}), 400

    sentence = data.get('sentence', '').strip()
    aspect = data.get('aspect', '').strip()
    model_type = data.get('model_type', 'lstm').strip()

    # 参数校验
    if not sentence:
        return jsonify({'error': '请输入句子'}), 400
    if not aspect:
        return jsonify({'error': '请输入方面词 (aspect)'}), 400
    if model_type not in ('lstm', 'bert_spc'):
        return jsonify({'error': '模型类型无效，可选: lstm, bert_spc'}), 400

    # 检查方面词是否在句子中
    if aspect.lower() not in sentence.lower():
        return jsonify({
            'warning': f'方面词 "{aspect}" 未在句子中找到，结果可能不准确',
            'result': _do_predict(sentence, aspect, model_type)
        })
    else:
        return jsonify({
            'result': _do_predict(sentence, aspect, model_type)
        })

def _do_predict(sentence, aspect, model_type):
    """执行推理"""
    try:
        if model_type == 'lstm':
            return predict_lstm(sentence, aspect)
        else:
            return predict_bert(sentence, aspect)
    except Exception as e:
        return {'error': str(e)}

# ============================================================
# 启动
# ============================================================
if __name__ == '__main__':
    print('Loading LSTM model...')
    load_lstm_model()
    print('LSTM model loaded OK')

    print('Loading BERT_SPC model...')
    load_bert_model()
    print('BERT_SPC model loaded OK')

    print(f'\nServer: http://127.0.0.1:5000')
    print(f'Device: {DEVICE}')
    app.run(host='127.0.0.1', port=5000, debug=False)
