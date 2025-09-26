"""
주의: 각 구현 블록에서 ".to()" 또는 ".cuda()"를 사용해서는 안 됩니다.
"""
import torch
from utils import Solver


def softmax_loss(x, y):
    """
    PyTorch를 이용한 softmax loss 구현입니다.
    """
    loss = torch.nn.functional.cross_entropy(x, y, reduction="mean")

    probs = torch.nn.functional.softmax(x, dim=1)
    N = x.shape[0]
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N

    return loss, dx


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        linear (fully-connected) layer에 대한 forward pass를 계산합니다.

        입력 x는 (N, d_1, ..., d_k) shape을 가지며 N개의 예제로 이루어진 minibatch입니다.
        각 예제 x[i]는 (d_1, ..., d_k) shape을 가집니다. 각 입력을 D = d_1 * ... * d_k 차원의 벡터로
        reshape한 다음, M 차원의 출력 벡터로 변환합니다.

        input:
        - x: (N, d_1, ..., d_k) shape의 입력 데이터를 포함하는 tensor
        - w: (D, M) shape의 weight tensor
        - b: (M,) shape의 bias tensor

        return (튜플):
        - out: (N, M) shape의 출력
        - cache: (x, w, b)
        """
        out = None
        ######################################################################
        # TODO: linear forward pass를 구현하세요. 결과는 out에 저장하세요.           #
        # 입력을 행으로 reshape 해야 합니다.                                       #
        ######################################################################
        pass
        ######################################################################
        #                        코드 끝                                       #
        ######################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        linear layer에 대한 backward pass를 계산합니다.

        입력:
        - dout: (N, M) shape의 Upstream derivative
        - cache: 튜플:
          - x: (N, d_1, ... d_k) shape의 입력 데이터
          - w: (D, M) shape의 weight
          - b: (M,) shape의 bias

        return (튜플):
        - dx: x에 대한 Gradient, (N, d1, ..., d_k) shape
        - dw: w에 대한 Gradient, (D, M) shape
        - db: b에 대한 Gradient, (M,) shape
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ##################################################
        # TODO: linear backward pass를 구현하세요.          #
        ##################################################
        pass
        ##################################################
        #                코드 끝                           #
        ##################################################
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        rectified linear units (ReLU) layer에 대한 forward pass를 계산합니다.

        입력:
        - x: 모든 shape의 입력 tensor

        return (튜플):
        - out: x와 동일한 shape의 출력 tensor
        - cache: x
        """
        out = None
        ###################################################
        # TODO: ReLU forward pass를 구현하세요.              #
        # 입력 tensor를 in-place 연산으로 변경해서는 안 됩니다.       #
        ###################################################
        pass
        ###################################################
        #                 코드 끝                           #
        ###################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        rectified linear units (ReLU) layer에 대한 backward pass를 계산합니다.

        입력:
        - dout: 모든 shape의 Upstream derivatives
        - cache: dout과 동일한 shape의 입력 x

        return:
        - dx: x에 대한 Gradient
        """
        dx, x = None, cache
        #####################################################
        # TODO: ReLU backward pass를 구현하세요.               #
        # 입력 tensor를 in-place 연산으로 변경해서는 안 됩니다.         #
        #####################################################
        pass
        #####################################################
        #                  코드 끝                            #
        #####################################################
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        linear 변환 후 ReLU를 수행하는 layer입니다.

        입력:
        - x: linear layer에 대한 입력
        - w, b: linear layer에 대한 weight

        return (튜플):
        - out: ReLU의 출력
        - cache: backward pass에 전달할 객체
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        linear-relu layer에 대한 backward pass
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class TwoLayerNet(object):
    """
    모듈식 layer 디자인을 사용하는 2-layer fully-connected 신경망입니다.
    입력 차원을 D, hidden 차원을 H로 가정하고 C개 class에 대해 분류를 수행합니다.
    아키텍처는 linear - relu - linear - softmax여야 합니다.
    이 class에서 gradient descent를 구현하지 않습니다.
    대신 최적화를 실행하는 별도의 Solver 객체와 상호 작용합니다.

    학습 가능한 모델 파라미터는 파라미터 이름을 torch tensor에 매핑하는 self.params 딕셔너리에 저장됩니다.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        """
        새로운 네트워크를 초기화합니다.

        입력:
        - input_dim: 입력의 크기를 나타내는 정수
        - hidden_dim: hidden layer의 크기를 나타내는 정수
        - num_classes: 분류할 class의 수를 나타내는 정수
        - weight_scale: weight의 무작위 초기화를 위한 표준 편차를 제공하는 scalar.
        - reg: L2 regularization strength를 제공하는 scalar.
        - dtype: torch 데이터 type 객체; 모든 계산은 이 데이터 type을 사용하여 수행됩니다.
          float는 빠르지만 정확도가 낮으므로 숫자 gradient 확인에는 double을 사용해야 합니다.
        - device: 계산에 사용할 장치. 'cpu' 또는 'cuda'
        """
        self.params = {}
        self.reg = reg

        ###################################################################
        # TODO: 2-layer 신경망의 weight와 bias을 초기화하세요.                   #
        # weight는 weight_scale과 동일한 표준편차를 갖는 0 중심의 가우시안 분포에서,   #
        # bias은 0으로 초기화해야 합니다.                                       #  
        # 모든 weight와 bias은 self.params 딕셔너리에 저장되어야 하며,             #
        # 첫 번째 layer의 weight와 bias은 'W1'과 'b1' 키를,                    #      
        # 두 번째 layer의 weight와 bias은 'W2'와 'b2' 키를 사용합니다.            #
        ###################################################################
        pass
        ###############################################################
        #                        코드 끝                             #
        ###############################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        데이터의 minibatch에 대한 loss과 gradient를 계산합니다.

        입력:
        - X: (N, d_1, ..., d_k) shape의 입력 데이터 tensor
        - y: (N,) shape의 int64 label tensor. y[i]는 X[i]의 label입니다.

        return:
        y가 None이면 모델의 test 모드이며, forward pass를 실행하고 다음을 return합니다.
        - scores: (N, C) shape의 tensor로, 분류 score를 제공합니다.
          scores[i, c]는 X[i]와 class c에 대한 분류 score입니다.
        y가 None이 아니면 학습 모드이며, forward 및 backward pass를 실행하고 다음 튜플을 return합니다.
        - loss: loss을 나타내는 scalar 값
        - grads: self.params와 동일한 키를 가진 딕셔너리로, 파라미터 이름을
          해당 파라미터에 대한 loss의 gradient에 매핑합니다.
        """
        scores = None
        #############################################################
        # TODO: 2-layer 신경망에 대한 forward pass를 구현하여,            #
        # X에 대한 class score를 계산하고 scores 변수에 저장하세요.          #
        #############################################################
        pass
        ##############################################################
        #                     코드 끝                                  #
        ##############################################################

        # y가 None이면 test 모드이므로 점수만 return합니다.
        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: 2-layer 신경망에 대한 backward pass를 구현하세요.                #
        # loss은 loss 변수에, gradient는 grads 딕셔너리에 저장하세요.             #
        # softmax를 사용하여 데이터 loss을 계산하고,                             #
        # grads[k]가 self.params[k]에 대한 gradient를 갖도록 하세요.            #
        # L2 regularization를 추가하는 것을 잊지 마세요!                         #
        #                                                                 #
        # 참고: 구현이 자동화된 test를 통과하려면                                 #
        # L2 regularization에 0.5 계수를 포함하지 않도록 하세요.                  #
        ###################################################################
        pass
        ###################################################################
        #                     코드 끝                                       #
        ###################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    여러 layer로 구성된 fully-connected 신경망입니다.
    L개의 layer이 있는 네트워크의 경우 아키텍처는 다음과 같습니다:

    {linear - relu} x (L - 1) - linear - softmax

    위의 TwoLayerNet과 유사하게, 학습 가능한 파라미터는 self.params 딕셔너리에 저장되며 Solver class를 사용하여 학습됩니다.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 reg=0.0, weight_scale=1e-2,
                 dtype=torch.float, device='cpu'):
        """
        새로운 FullyConnectedNet을 초기화합니다.

        입력:
        - hidden_dims: 각 hidden layer의 크기를 제공하는 정수 리스트.
        - input_dim: 입력의 크기를 나타내는 정수.
        - num_classes: 분류할 class의 수를 나타내는 정수.
        - reg: L2 regularization strength를 제공하는 scalar.
        - weight_scale: weight의 무작위 초기화를 위한 표준 편차를 제공하는 scalar.
        - dtype: torch 데이터 type 객체; 모든 계산은 이 데이터 type을 사용하여 수행됩니다.
          float는 빠르지만 정확도가 낮으므로 숫자 gradient 확인에는 double을 사용해야 합니다.
        - device: 계산에 사용할 장치. 'cpu' 또는 'cuda'
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: 네트워크의 파라미터를 초기화하고, 모든 값을 self.params 딕셔너리에 저장하세요.     #
        # 첫 번째 layer의 weight와 bias은 W1과 b1에, 두 번째 layer은 W2와 b2 등을 사용하세요. #
        # weight는 weight_scale과 동일한 표준편차를 갖는 0 중심의 정규 분포에서 초기화해야 합니다. #
        # bias은 0으로 초기화해야 합니다.                                                #
        ############################################################################
        pass
        #######################################################################
        #                         코드 끝                                       #
        #######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        fully-connected net의 loss과 gradient를 계산합니다.
        입력/출력: 위의 TwoLayerNet과 동일합니다.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        scores = None
        ##################################################################
        # TODO: fully-connected net에 대한 forward pass를 구현하여,           #
        # X에 대한 class score를 계산하고 scores 변수에 저장하세요.               #
        ##################################################################
        pass
        #################################################################
        #                      코드 끝                                    #
        #################################################################

        # test 모드이면 여기서 return합니다.
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #####################################################################
        # TODO: fully-connected net에 대한 backward pass를 구현하세요.            #
        # loss은 loss 변수에, gradient는 grads 딕셔너리에 저장하세요.                #
        # softmax를 사용하여 데이터 loss을 계산하고,                                #
        # grads[k]가 self.params[k]에 대한 gradient를 갖도록 하세요.               # 
        # L2 regularization를 추가하는 것을 잊지 마세요!                           #
        #                                                                   #
        # 참고: 구현이 자동화된 test를 통과하려면                                   #
        # L2 regularization에 0.5 계수를 포함하지 않도록 하세요.                    #
        #####################################################################
        pass
        ###########################################################
        #                   코드 끝                                 #
        ###########################################################

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    #############################################################
    # TODO: Solver 인스턴스를 생성하세요.                             #
    #############################################################
    solver = None
    pass
    ##############################################################
    #                    코드 끝                                  #
    ##############################################################
    return solver


def get_three_layer_network_params():
    ###############################################################
    # TODO: weight_scale과 learning_rate를 변경해 training accuracy를 #
    # 100% 달성할 수 있도록 합니다.                                     #
    ###############################################################
    weight_scale = 1e-2   # Experiment with this!
    learning_rate = 1e-4  # Experiment with this!
    ################################################################
    #                             코드 끝                           #
    ################################################################
    return weight_scale, learning_rate


def get_five_layer_network_params():
    ###############################################################
    # TODO: weight_scale과 learning_rate를 변경해 training accuracy를 #
    # 100% 달성할 수 있도록 합니다.                                     #
    ###############################################################
    weight_scale = 1e-2   # Experiment with this!
    learning_rate = 1e-4  # Experiment with this!
    ################################################################
    #                       코드 끝                                 #
    ################################################################
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    vanilla stochastic gradient descent를 수행합니다.
    config 형식:
    - learning_rate: scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    모멘텀을 사용한 stochastic gradient descent를 수행합니다.
    config 형식:
    - learning_rate: scalar learning rate.
    - momentum: 0과 1 사이의 scalar로 모멘텀 값을 제공합니다.
      momentum = 0으로 설정하면 sgd가 됩니다.
    - velocity: w 및 dw와 동일한 모양의 numpy 배열로,
      gradient의 이동 평균을 저장하는 데 사용됩니다.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))

    next_w = None
    ##################################################################
    # TODO: 모멘텀 업데이트 공식을 구현하세요.                               #
    # 업데이트된 값은 next_w 변수에 저장하세요.                               #
    # 또한 velocity v를 사용하고 업데이트해야 합니다.                         #
    ##################################################################
    pass
    ###################################################################
    #                           코드 끝                                #
    ###################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    config 형식:
    - learning_rate: scalar learning rate.
    - decay_rate: 제곱 gradient 캐시의 decay rate(감쇠율)을 나타내는 0과 1 사이의 scalar.
    - epsilon: 0으로 나누는 것을 방지하기 위해 스무딩에 사용되는 작은 scalar.
    - cache: gradient의 2차 모멘트의 이동 평균.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: RMSprop 업데이트 공식을 구현하고, w의 다음 값을 next_w 변수에 저장하세요.      #
    # config['cache']에 저장된 캐시 값을 업데이트하는 것을 잊지 마세요.                  #
    ###########################################################################
    pass
    ###########################################################################
    #                             코드 끝                                       #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    config 형식:
    - learning_rate: scalar learning rate.
    - beta1: gradient의 1차 모멘트 이동 평균의 decay rate(감쇠율).
    - beta2: gradient의 2차 모멘트 이동 평균의 decay rate(감쇠율).
    - epsilon: 0으로 나누는 것을 방지하기 위해 스무딩에 사용되는 작은 scalar.
    - m: gradient의 이동 평균.
    - v: 제곱 gradient의 이동 평균.
    - t: 반복 횟수.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    ##########################################################################
    # TODO: Adam 업데이트 공식을 구현하고, w의 다음 값을 next_w 변수에 저장하세요.         #
    # config에 저장된 m, v, t 변수를 업데이트하는 것을 잊지 마세요.                     #
    #                                                                        #
    # 참고: 계산에 t를 사용하기 전에 t를 수정하세요.                                   #
    ##########################################################################
    pass
    #########################################################################
    #                              코드 끝                                    #
    #########################################################################

    return next_w, config