import torch
from torch import nn, Tensor
from typing import Tuple
import unicodedata
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BPTT = 35

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

# 将样本划分为batch
def batchify(data: Tensor, bsz: int) -> Tensor:
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(DEVICE)

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    seq_len = min(BPTT, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

# output是一个概率向量，每个元素代表每个分类属于的概率
def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# 绘制混淆矩阵图，并返回混淆矩阵
# trues和predicts列表分别存放真实与预测的分类序号
# all_categories为类别的文字说明（只用于绘图）

def plot_confusion_matrix(category_size: int, tures: Tensor, predicts: Tensor, all_categories: list = []) -> Tensor:
    confusion = torch.zeros(category_size, category_size)

    assert tures.size(0) == predicts.size(0), "true's size must equal to predicts's size"

    for (true_index, predict_index) in zip(tures, predicts):
        assert true_index < category_size and predict_index < category_size, "true index or predict index is out of category size"
        confusion[true_index][predict_index] += 1

    for i in range(category_size):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

    return confusion