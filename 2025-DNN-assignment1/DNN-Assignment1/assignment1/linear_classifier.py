"""
주의: 각 구현 블록에서 ".to()" 또는 ".cuda()"를 사용해서는 안 됩니다.
"""
import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional


# 이 class를 편집/수정하지 마세요.
class LinearClassifier:
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        """
        loss function와 그 gradient를 계산합니다.
        subclass에서 이 메서드를 오버라이드합니다.

        입력:
        - W: 모델의 (학습된) weight를 포함하는 (D, C) shape의 torch tensor.
        - X_batch: N개의 데이터 포인트를 포함하는 minibatch. 각 포인트는 D차원이며 (N, D) shape의 torch tensor.
        - y_batch: minibatch에 대한 label을 포함하는 (N,) shape의 torch tensor.
        - reg: (float) regularization strength.

        반환: 다음을 포함하는 튜플:
        - 단일 float 값의 loss
        - self.W에 대한 gradient; W와 동일한 shape의 tensor
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class Softmax(LinearClassifier):
    """Softmax + cross-entropy loss function를 사용하는 subclass"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    학습 데이터에서 batch_size개의 element와 해당 label을 샘플링하여
    이번 gradient descent(경사 하강법) 단계에서 사용합니다.
    """
    X_batch = None
    y_batch = None
    #########################################################################
    # TODO: 데이터를 X_batch에, 해당 label을 y_batch에 저장하세요.                  #
    # 샘플링 후, X_batch는 (batch_size, dim) shape, y_batch는 (batch_size,)     #
    # shape이어야 합니다.                                                       #
    #                                                                       #
    # 힌트: torch.randint를 사용하여 인덱스를 생성하세요.                            #
    #########################################################################
    pass
    #########################################################################
    #                       코드 끝                                          #
    #########################################################################
    return X_batch, y_batch


def train_linear_classifier(
    loss_func: Callable,
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Gradient descent 기법을 사용하여 이 linear classifier(linear classifier)를 학습합니다.

    입력:
    - loss_func: 학습 시 사용할 loss function. W, X, y, reg를 입력으로 받고
      (loss, dW) 튜플을 출력해야 합니다.
    - W: 분류기의 초기 weight를 제공하는 (D, C) shape의 torch tensor.
      W가 None이면 이 함수에서 초기화됩니다.
    - X: 학습 데이터를 포함하는 (N, D) shape의 torch tensor.
      N개의 학습 샘플이 있으며 각각 D차원입니다.
    - y: 학습 label을 포함하는 (N,) shape의 torch tensor;
      y[i] = c는 X[i]가 C개의 클래스 중 0 <= c < C label을 가짐을 의미합니다.
    - learning_rate: (float) 최적화를 위한 learning rate.
    - reg: (float) regularization strength.
    - num_iters: (integer) 최적화 시 수행할 단계 수.
    - batch_size: (integer) 각 단계에서 사용할 학습 샘플의 수.
    - verbose: (boolean) True이면 최적화 중 진행 상황을 출력합니다.

    반환: 다음을 포함하는 튜플:
    - W: 최적화 종료 시 weight 행렬의 최종 값
    - loss_history: 각 학습 반복에서의 loss 값을 제공하는 파이썬 스칼라 리스트.
    """
    # y가 0...K-1 값을 가진다고 가정 (K는 클래스 수)
    num_train, dim = X.shape
    if W is None:
        # W를 초기화
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(
            dim, num_classes, device=X.device, dtype=X.dtype
        )
    else:
        num_classes = W.shape[1]

    # gradient descent 실행하여 W를 최적화
    loss_history = []
    for it in range(num_iters):
        # TODO: sample_batch 함수 구현
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # loss과 gradient 평가
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        # 파라미터 업데이트 수행
        #########################################################################
        # TODO:                                                                 #
        # gradient와 learning rate을 사용하여 weight를 업데이트하세요.                  #
        #########################################################################
        pass
        #########################################################################
        #                       코드 끝                                         #
        #########################################################################

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    """
    이 linear classifier의 학습된 weight를 사용하여 데이터 포인트의 label을 예측합니다.

    입력:
    - W: 모델의 weight를 포함하는 (D, C) shape의 torch tensor.
    - X: 학습 데이터를 포함하는 (N, D) shape의 torch tensor.
      N개의 학습 샘플이 있으며 각각 D차원입니다.

    반환:
    - y_pred: X의 각 element에 대한 예측 label을 제공하는 (N,) shape의 torch int64 tensor.
      y_pred의 각 element는 0과 C - 1 사이여야 합니다.
    """
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    ###########################################################################
    # TODO:                                                                   #
    # 이 메서드를 구현하세요. 예측된 label을 y_pred에 저장하세요.                        #
    ###########################################################################
    pass
    ###########################################################################
    #                           코드 끝                                         #
    ###########################################################################
    return y_pred



def softmax_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function을 반복문을 사용해서 구현합니다.
    주의사항: W에 대한 regularization를 구현할 때, regularization 항에 1/2을 곱하지 마세요.

    입력은 D차원, C개의 클래스가 있으며, N개의 예제로 구성된 minibatch에 대해 연산합니다.

    입력:
    - W: weight를 포함하는 (D, C) shape의 torch tensor.
    - X: 데이터의 minibatch를 포함하는 (N, D) shape의 torch tensor.
    - y: 학습 label을 포함하는 (N,) shape의 torch tensor;
      y[i] = c는 X[i]의 label이 c임을 의미하며, 0 <= c < C 입니다.
    - reg: (float) regularization strength

    반환: 다음을 포함하는 튜플:
    - 단일 float 값의 loss
    - weight W에 대한 loss의 gradient; W와 동일한 shape의 tensor
    """
    # loss과 gradient를 0으로 초기화합니다.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: 반복문을 사용하여 softmax loss과 gradient를 계산하세요.                      #
    # loss은 loss에, gradient는 dW에 저장하세요.                                     #                   
    # 구현 시 주의하지 않으면 수치적 불안정성(numerical instability)에 빠질 수 있습니다.      #
    # http://cs231n.github.io/linear-classify/ 의 Numeric Stability 항목 확인)     #
    #############################################################################
    pass
    #############################################################################
    #                          코드 끝                                            #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function의 vectorized 버전 (반복문을 사용하지 않습니다).
    주의사항: W에 대한 regularization를 구현할 때, regularization 항에 1/2을 곱하지 마세요.

    입력과 출력은 softmax_loss_naive와 동일합니다.
    """
    # loss과 gradient를 0으로 초기화합니다.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: 반복문 사용 없이 softmax loss과 gradient를 계산하세요.                      #
    # loss은 loss에, gradient는 dW에 저장하세요.                                     #                   
    # 구현 시 주의하지 않으면 수치적 불안정성(numerical instability)에 빠질 수 있습니다.      #
    # http://cs231n.github.io/linear-classify/ 의 Numeric Stability 항목 확인)     #
    #############################################################################
    num_train = X.shape[0]
    #############################################################################
    #                          코드 끝                                            #
    #############################################################################

    return loss, dW


def softmax_get_search_params():
    """
    Softmax 모델에 대해 하이퍼파라미터 탐색을 위한 하이퍼파라미터 후보를 return합니다.
    각 파라미터에 대해 적어도 두 개의 후보를 제공해야 하며, 총 그리드 검색 조합은 25개 미만이어야 합니다.

    반환:
    - learning_rates: learning rate 후보, 예: [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strength 후보, 예: [1e0, 1e1, ...]
    """
    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO: 자신만의 하이퍼파라미터 리스트를 추가하세요.                                 #
    ###########################################################################
    pass
    ###########################################################################
    #                           코드 끝                                       #
    ###########################################################################

    return learning_rates, regularization_strengths


def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    """
    단일 LinearClassifier 인스턴스를 학습시키고 학습된 인스턴스를 학습/검증 정확도와 함께 return 합니다.

    입력:
    - cls (LinearClassifier): 새로 생성된 LinearClassifier 인스턴스.
                              이 인스턴스에 대해 학습/테스트를 수행해야 합니다.
    - data_dict (dict): 분류기 학습을 위한 ['X_train', 'y_train', 'X_val', 'y_val']
                        키를 포함하는 딕셔너리.
    - lr (float): 모델 인스턴스 학습을 위한 학습률 파라미터.
    - reg (float): 모델 인스턴스 학습을 위한 정규화 weight.
    - num_iters (int, optional): 학습할 반복 횟수.

    반환:
    - cls (LinearClassifier): num_iter 횟수만큼 (['X_train', 'y_train'], lr, reg)로
                              학습된 LinearClassifier 인스턴스.
    - train_acc (float): 모델의 학습 정확도.
    - val_acc (float): 모델의 테스트 정확도.
    """
    train_acc = 0.0  # accuracy(정확도)는 맞게 분류된 데이터 포인트의 비율입니다.
    val_acc = 0.0
    ###########################################################################
    # TODO:                                                                   #
    # 학습 데이터셋에서 모델을 학습시키고 학습 및 테스트셋에서 정확도를 계산하는 코드를 작성하세요.  #
    ###########################################################################
    pass
    ############################################################################
    #                            코드 끝                                        #
    ############################################################################

    return cls, train_acc, val_acc