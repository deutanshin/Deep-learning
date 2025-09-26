"""
주의: 각 구현 블록에서 ".to()" 또는 ".cuda()"를 사용해서는 안 됩니다.
"""
import torch
import random
import statistics
from linear_classifier import sample_batch
from typing import Dict, List, Callable, Optional


# 이 class를 편집/수정하지 마세요.
class TwoLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        std: float = 1e-4,
    ):
        """
        모델을 초기화합니다. weight는 작은 무작위 값으로 초기화되고 bias는 0으로 초기화됩니다.
        weight와 bias은 self.params 변수에 저장되며, 이 변수는 다음 키를 가진 딕셔너리입니다:

        W1: 첫 번째 layer weight; (D, H) shape
        b1: 첫 번째 layer bias; (H,) shape
        W2: 두 번째 layer weight; (H, C) shape
        b2: 두 번째 layer bias; (C,) shape

        입력:
        - input_size: 입력 데이터의 차원 D.
        - hidden_size: hidden layer의 뉴런 수 H.
        - output_size: class 수 C.
        - dtype: 선택 사항, 각 초기 weight 파라미터의 데이터 타입
        - device: 선택 사항, weight 파라미터가 GPU에 있는지 CPU에 있는지 여부
        - std: 선택 사항, 초기 weight scaler.
        """
        random.seed(0)
        torch.manual_seed(0)

        self.params = {}
        self.params["W1"] = std * torch.randn(
            input_size, hidden_size, dtype=dtype, device=device
        )
        self.params["b1"] = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.params["W2"] = std * torch.randn(
            hidden_size, output_size, dtype=dtype, device=device
        )
        self.params["b2"] = torch.zeros(output_size, dtype=dtype, device=device)

    def loss(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        reg: float = 0.0,
    ):
        return nn_forward_backward(self.params, X, y, reg)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        return nn_train(
            self.params, nn_forward_backward, nn_predict, X, y,
            X_val, y_val, learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )

    def predict(self, X: torch.Tensor):
        return nn_predict(self.params, nn_forward_backward, X)

    def save(self, path: str):
        torch.save(self.params, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint
        if len(self.params) != 4:
            raise Exception("Failed to load your checkpoint")

        for param in ["W1", "b1", "W2", "b2"]:
            if param not in self.params:
                raise Exception("Failed to load your checkpoint")


def nn_forward_pass(params: Dict[str, torch.Tensor], X: torch.Tensor):
    """
    신경망 구현의 첫 번째 단계: 네트워크의 forward pass를 실행하여
    hidden layer feature와 분류 score를 계산합니다.
    네트워크 아키텍처는 다음과 같아야 합니다:

    FC layer -> ReLU (hidden) -> FC layer (score)

    주의 사항: torch.relu와 torch.nn 연산을 사용하지 않습니다.

    입력:
    - params: 모델의 weight를 저장하는 torch tensor의 딕셔너리.
      다음 키와 shape을 가져야 합니다.
          W1: 첫 번째 layer weight; (D, H) shape
          b1: 첫 번째 layer bias; (H,) shape
          W2: 두 번째 layer weight; (H, C) shape
          b2: 두 번째 layer bias; (C,) shape
    - X: (N, D) shape의 입력 데이터. 각 X[i]는 학습 샘플입니다.

    return: 다음을 포함하는 튜플:
    - scores: X에 대한 분류 score를 제공하는 (N, C) shape의 tensor
    - hidden: 각 입력 값에 대한 hidden layer 표현을 제공하는 (N, H) shape의 tensor (ReLU 이후).
    """
    # params 딕셔너리에서 변수 언패킹
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    # forward pass 계산
    hidden = None
    scores = None
    ############################################################################
    # TODO: 입력에 대한 class score를 계산하는 forward pass를 수행하세요.               #
    # 결과를 scores 변수에 저장하며 이 변수는 (N, C) shape의 tensor여야 합니다.           #
    ############################################################################
    pass
    ###########################################################################
    #                             코드 끝                                     #
    ###########################################################################

    return scores, hidden


def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0
):
    """
    2-layer fully connected 신경망에 대한 loss과 gradient를 계산합니다.
    loss과 gradient를 구현할 때, batch 크기로 loss/gradient를 scaling하는 것을 잊지 마세요.

    입력: 첫 두 파라미터(params, X)는 nn_forward_pass와 동일합니다.
    - params: 모델의 weight를 저장하는 torch tensor의 딕셔너리.
      다음 키와 shape을 가져야 합니다.
          W1: 첫 번째 layer weight; (D, H) shape
          b1: 첫 번째 layer bias; (H,) shape
          W2: 두 번째 layer weight; (H, C) shape
          b2: 두 번째 layer bias; (C,) shape
    - X: (N, D) shape의 입력 데이터. 각 X[i]는 학습 샘플입니다.
    - y: 학습 label 벡터. y[i]는 X[i]의 label이며,
      각 y[i]는 0 <= y[i] < C 범위의 정수입니다. 이 파라미터는 선택 사항입니다.
      전달되지 않으면 score만 return하고, 전달되면 loss과 gradient를 return합니다.
    - reg: regularization strength.

    return:
    y가 None이면, scores[i, c]가 입력 X[i]에 대한 class c의 score인
    (N, C) shape의 tensor scores를 return합니다.

    y가 None이 아니면, 다음을 포함하는 튜플을 return합니다:
    - loss: 이 학습 샘플 batch에 대한 loss (데이터 loss 및 regularization loss).
    - grads: 파라미터 이름을 해당 파라미터의 loss 함수에 대한 gradient에
      매핑하는 딕셔너리; self.params와 동일한 키를 가집니다.
    """
    # params 딕셔너리에서 변수 언패킹
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    scores, h1 = nn_forward_pass(params, X)
    # target y가 주어지지 않으면 함수는 여기서 종료합니다.
    if y is None:
        return scores

    # loss 계산
    loss = None
    ############################################################################
    # TODO: nn_forward_pass의 결과를 기반으로 loss을 계산하세요.                       #
    # 여기에는 데이터 loss과 W1 및 W2에 대한 L2 regularization가 모두 포함되어야 합니다.    #
    # 결과를 scalar여야 하고 이 값은 loss 변수에 저장하되어야 합니다.                       #
    # Loss function은 Softmax cross-entropy loss를 사용합니다.                     #
    # W에 대한 regularization를 구현할 때, regularization 항에 1/2을 곱하지 마세요       #
    ############################################################################
    pass
    ###########################################################################
    #                             코드 끝                                     #
    ###########################################################################

    # backward pass: gradient 계산
    grads = {}
    ###########################################################################
    # TODO: backward pass를 계산하여 weight와 bias의 gradient를 계산하세요.          #
    # 결과를 grads 딕셔너리에 저장하세요. 예를 들어, grads['W1']은 W1에 대한              #
    # gradient를 저장해야 하며, 동일한 크기의 tensor여야 합니다.                        #
    ###########################################################################
    pass
    ###########################################################################
    #                             코드 끝                                     #
    ###########################################################################

    return loss, grads


def nn_train(
    params: Dict[str, torch.Tensor],
    loss_func: Callable,
    pred_func: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float = 1e-3,
    learning_rate_decay: float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    SGD를 사용하여 이 신경망을 학습합니다.

    입력:
    - params: 모델의 weight를 저장하는 torch tensor의 딕셔너리.
      다음 키와 shape을 가져야 합니다.
          W1: 첫 번째 layer weight; (D, H) shape
          b1: 첫 번째 layer bias; (H,) shape
          W2: 두 번째 layer weight; (H, C) shape
          b2: 두 번째 layer bias; (C,) shape
    - loss_func: loss과 gradient를 계산하는 loss 함수.
      다음 입력을 받습니다:
      - params: nn_train의 입력과 동일
      - X_batch: (B, D) shape의 입력 minibatch
      - y_batch: X_batch에 대한 실제 label
      - reg: nn_train의 입력과 동일
      그리고 다음 튜플을 return합니다:
        - loss: minibatch에 대한 scalar loss
        - grads: 파라미터 이름을 해당 파라미터에 대한 loss의 gradient에 매핑하는 딕셔너리.
    
    - pred_func: 예측 함수
    - X: 학습 데이터를 제공하는 (N, D) shape의 torch tensor.
    - y: 학습 label을 제공하는 (N,) shape의 torch tensor;
      y[i] = c는 X[i]가 class c를 가짐을 의미하며, 0 <= c < C 입니다.
    - X_val: 테스트 데이터를 제공하는 (N_val, D) shape의 torch tensor.
    - y_val: 테스트 label을 제공하는 (N_val,) shape의 torch tensor.
    - learning_rate: 최적화를 위한 scalar learning rate.
    - learning_rate_decay: 각 epoch 후 learning rate을 감소시키는 데 사용되는 scalar 계수.
    - reg: scalar regularization strength.
    - num_iters: 최적화 시 수행할 단계 수.
    - batch_size: 단계별로 사용할 학습 예제의 수.
    - verbose: boolean; True이면 최적화 중 진행 상황을 출력합니다.

    return: 학습 과정에 대한 통계를 제공하는 딕셔너리
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # SGD를 사용하여 self.model의 파라미터를 최적화합니다.
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # 현재 minibatch를 사용하여 loss과 gradient를 계산합니다.
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())

        #########################################################################
        # TODO: grads 딕셔너리의 gradient를 사용하여 self.params 딕셔너리에 저장된 네트워크의 #
        # 파라미터를 SGD 기법을 사용하여 업데이트하세요.                                   #      
        # 위에서 정의된 grads 딕셔너리에 저장된 gradient를 사용해야 합니다.                  #
        #########################################################################
        pass
        #########################################################################
        #                             코드 끝                                   #
        #########################################################################

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss.item()))

        # 매 epoch마다 학습 및 테스트 accuracy를 확인하고 learning rate을 감소시킵니다.
        if it % iterations_per_epoch == 0:
            # accuracy 확인
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # learning rate 감소
            learning_rate *= learning_rate_decay

    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
    }


def nn_predict(
    params: Dict[str, torch.Tensor], loss_func: Callable, X: torch.Tensor
):
    """
    이 2-layer 네트워크의 학습된 weight를 사용하여 데이터의 label을 예측합니다.
    각 데이터에 대해 C개의 class 각각에 대한 score를 예측하고,
    각 데이터를 가장 높은 score를 가진 class에 할당합니다.

    입력:
    - params: 모델의 weight를 저장하는 torch tensor의 딕셔너리.
      다음 키와 shape을 가져야 합니다.
          W1: 첫 번째 layer weight; (D, H) shape
          b1: 첫 번째 layer bias; (H,) shape
          W2: 두 번째 layer weight; (H, C) shape
          b2: 두 번째 layer bias; (C,) shape
    - loss_func: loss과 gradient를 계산하는 loss 함수
    - X: 분류할 N개의 D차원 데이터를 제공하는 (N, D) shape의 torch tensor.

    return:
    - y_pred: X의 각 요소에 대한 예측 label을 제공하는 (N,) shape의 torch tensor.
      모든 i에 대해, y_pred[i] = c는 X[i]가 class c를 가질 것으로 예측됨을 의미하며,
      0 <= c < C 입니다.
    """
    y_pred = None

    ###########################################################################
    # TODO: 이 함수를 구현하세요. 매우 간단해야 합니다!                                 #
    ###########################################################################
    pass
    ###########################################################################
    #                              코드 끝                                    #
    ###########################################################################

    return y_pred