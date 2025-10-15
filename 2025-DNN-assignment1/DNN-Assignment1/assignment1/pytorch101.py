import torch
from typing import List, Tuple
from torch import Tensor


def create_sample_tensor() -> Tensor:
    """
    (3, 2) shape의 torch tensor를 return 합니다. 이 tensor는 0으로 채워져 있으며
    (0, 1) element는 10,
    (1, 0) element는 100으로 설정됩니다.

    return:
        위에서 설명한 (3, 2) shape의 tensor.
    """
    x = None
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    ##########################################################################
    x = torch.tensor([[0,10],[100,0],[0,0]])
    ##########################################################################
    #                            코드 끝                                       #
    ##########################################################################
    return x


def mutate_tensor(
    x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:
    """
    인덱스와 값에 따라 tensor x를 변경합니다.
    indices는 정수 인덱스의 리스트 [(i0, j0), (i1, j1), ... ]이고, values는
    [v0, v1, ...] 값의 리스트입니다. 이 함수는 다음을 설정하여 x를 변경해야 합니다:

    x[i0, j0] = v0
    x[i1, j1] = v1

    동일한 인덱스 쌍이 인덱스에 여러 번 나타나면 x를
    마지막 값으로 설정해야 합니다.

    Args:
        x: (H, W) shape의 tensor
        indices: N개의 튜플 [(i0, j0), (i1, j1), ..., ] 리스트
        values: N개의 값 [v0, v1, ...] 리스트

    Returns:
        입력 tensor x
    """
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    ##########################################################################
    for i in range(3):
        x[indices[i][0]][indices[i][1]] = values[i]
    ##########################################################################
    #                            코드 끝                                       #
    ##########################################################################
    return x


def count_tensor_elements(x: Tensor) -> int:
    """
    tensor x의 scalar element 수를 셉니다.

    예를 들어, (10,) shape의 tensor는 10개의 element를 가집니다.
    (3, 4) shape의 tensor는 12개의 element를 가집니다.
    (2, 3, 4) shape의 tensor는 24개의 element를 가집니다.

    torch.numel 또는 x.numel 함수를 사용할 수 없습니다.
    입력 tensor는 수정되어서는 안 됩니다.

    Args:
        x: 모든 shape의 tensor

    Returns:
        num_elements: x의 scalar element 수를 나타내는 정수
    """
    num_elements = None
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    #   내장 함수인 torch.numel(x) 또는 x.numel()을 사용할 수 없습니다.               #       
    ##########################################################################
    num_elements = 1
    for i in x.shape:
        num_elements *= i
    ##########################################################################
    #                            코드 끝                                       #
    ##########################################################################
    return num_elements


def create_tensor_of_pi(M: int, N: int) -> Tensor:
    """
    값이 3.14로 채워진 (M, N) shape의 tensor를 return합니다.

    Args:
        M, N: 생성할 tensor의 shape을 나타내는 양의 정수

    Returns:
        x: 값이 3.14로 채워진 (M, N) shape의 tensor
    """
    x = None
    ##########################################################################
    #         TODO: 여기에 코드를 작성하세요. 코드는 한 줄로 작성해야 합니다.             #
    ##########################################################################
    x = torch.zeros(M, N, dtype=torch.float32)
    for i in range(M):
        for j in range(N):
            x[i, j] = 3.14
    ##########################################################################
    #                            코드 끝                                       #
    ##########################################################################
    return x


def multiples_of_ten(start: int, stop: int) -> Tensor:
    """
    start와 stop 사이의 10의 모든 배수를 (순서대로) 포함하는
    dtype torch.float64의 tensor를 return합니다.
    이 범위에 10의 배수가 없으면 (0,) shape의 빈 tensor를 return합니다.

    Args:
        start: 생성할 범위의 시작.
        stop: 생성할 범위의 끝 (stop >= start).

    Returns:
        x: start와 stop 사이의 10의 배수를 제공하는 float64 tensor
    """
    assert start <= stop
    x = None
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    ##########################################################################
    s = None

    if start % 10 == 0:
        s = start
    else:
        s = (start // 10 + 1) * 10

    if s > stop:
        return torch.tensor([], dtype=torch.float64)

    x = torch.arange(s, stop+1, 10, dtype=torch.float64)       
    ##########################################################################
    #                            코드 끝                                       #
    ##########################################################################
    return x


def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    주어진 2차원 tensor x에서 여러 sub-tensor를 추출하여 return합니다.
    각 tensor는 단일 슬라이스 인덱싱 연산으로 생성되어야 합니다.
    입력 tensor는 수정되어서는 안 됩니다.

    Args:
        x: (M, N) shape의 tensor -- M 행, N 열, M >= 3, N >= 5.

    Returns:
        다음 튜플:
        - last_row: x의 마지막 행을 나타내는 (N,) shape의 tensor. 1차원 tensor여야 합니다.
        - third_col: x의 세 번째 열을 나타내는 (M, 1) shape의 tensor. 2차원 tensor여야 합니다.
        - first_two_rows_three_cols: x의 첫 두 행과 첫 세 열의 데이터를 나타내는 (2, 3) shape의 tensor.
        - even_rows_odd_cols: x의 짝수 행과 홀수 열에 있는 element를 포함하는 2차원 tensor.
    """
    assert x.shape[0] >= 3
    assert x.shape[1] >= 5
    last_row = None
    third_col = None
    first_two_rows_three_cols = None
    even_rows_odd_cols = None
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    ##########################################################################
    last_row = x[x.shape[0] - 1, :]
    third_col = x[:, 2:3]
    first_two_rows_three_cols = x[:2, :3]
    even_rows_odd_cols = x[::2, 1::2]
    ##########################################################################
    #                            코드 끝                                   #
    ##########################################################################
    out = (
        last_row,
        third_col,
        first_two_rows_three_cols,
        even_rows_odd_cols,
    )
    return out


def slice_assignment_practice(x: Tensor) -> Tensor:
    """
    M >= 4, N >= 6인 (M, N) shape의 2차원 tensor가 주어졌을 때,
    첫 4개 행과 6개 열을 다음과 같이 변경합니다:

    [0 1 2 2 2 2]
    [0 1 2 2 2 2]
    [3 4 3 4 5 5]
    [3 4 3 4 5 5]

    참고: 입력 tensor shape은 (4, 6)으로 고정되지 않습니다.

    구현은 다음을 따라야 합니다:
    - tensor x를 제자리에서 변경하고 return해야 합니다.
    - 첫 4개 행과 첫 6개 열만 수정해야 합니다. 다른 모든 element는 변경되지 않아야 합니다.
    - tensor의 슬라이스에 정수를 할당하는 슬라이스 할당 연산만 사용하여 tensor를 변경할 수 있습니다.
    - 원하는 결과를 얻기 위해 <= 6개의 슬라이싱 연산을 사용해야 합니다.

    Args:
        x: M >= 4, N >= 6인 (M, N) shape의 tensor

    Returns:
        x
    """
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    ##########################################################################
    x[:2, 2:6] = 2
    x[2:4, :6:2] = 3
    x[2:4, 1:6:2] = 4
    x[2:4, 4:6] = 5
    x[:2, :1] = 0
    x[:2, 1:2] = 1
    ##########################################################################
    #                            코드 끝                                       #
    ##########################################################################
    return x


def shuffle_cols(x: Tensor) -> Tensor:
    """
    아래 설명된 대로 입력 tensor의 열 순서를 변경합니다.

    구현은 단일 정수 배열 인덱싱 연산을 사용하여 출력 tensor를 구성해야 합니다.
    입력 tensor는 수정되어서는 안 됩니다.

    Args:
        x: N >= 3인 (M, N) shape의 tensor

    Returns:
        (M, 4) shape의 tensor y:
        - y의 첫 두 열은 x의 첫 번째 열의 복사본입니다.
        - y의 세 번째 열은 x의 세 번째 열과 동일합니다.
        - y의 네 번째 열은 x의 두 번째 열과 동일합니다.
    """
    y = None
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요.                       #
    ##########################################################################
    y = torch.zeros(x.shape[0], 4)
    idx = [0, 0, 2, 1]
    y = x[:, idx]
    ##########################################################################
    #                            코드 끝                                      #
    ##########################################################################
    return y


def reverse_rows(x: Tensor) -> Tensor:
    """
    입력 tensor의 행을 뒤집습니다.

    구현은 단일 정수 배열 인덱싱 연산을 사용하여 출력 tensor를 구성해야 합니다.
    입력 tensor는 수정되어서는 안 됩니다.

    구현 시 torch.flip을 사용할 수 없습니다.

    Args:
        x: (M, N) shape의 tensor

    Returns:
        y: (M, N) shape의 tensor. x와 동일하지만 행이 뒤집혔습니다.
           y의 첫 번째 행은 x의 마지막 행과 같아야 하고,
           y의 두 번째 행은 x의 끝에서 두 번째 행과 같아야 합니다.
    """
    y = None
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    ##########################################################################
    idx = torch.arange(x.shape[0] - 1, -1, -1)
    y = x[idx, :]
    ##########################################################################
    #                            코드 끝                                      #
    ##########################################################################
    return y


def take_one_elem_per_col(x: Tensor) -> Tensor:
    """
    아래 설명된 대로 입력 tensor의 각 열에서 하나의 element를 선택하여 새 tensor를 구성합니다.

    입력 tensor는 수정되어서는 안 되며, 단일 인덱싱 연산만 사용하여
    입력 tensor의 데이터에 액세스해야 합니다.

    Args:
        x: M >= 4, N >= 3인 (M, N) shape의 tensor.

    Returns:
        (3,) shape의 tensor y:
        - y의 첫 번째 element는 x의 첫 번째 열의 두 번째 element입니다.
        - y의 두 번째 element는 x의 두 번째 열의 첫 번째 element입니다.
        - y의 세 번째 element는 x의 세 번째 열의 네 번째 element입니다.
    """
    y = None
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    ##########################################################################
    col = [0, 1, 2]
    row = [1, 0, 3]
    y = x[row, col]
    ##########################################################################
    #                            코드 끝                                      #
    ##########################################################################
    return y


def reshape_practice(x: Tensor) -> Tensor:
    """
    (24,) shape의 입력 tensor가 주어졌을 때, (3, 8) shape의 재구성된 tensor y를 return합니다.

    y = [[x[0], x[1], x[2],  x[3],  x[12], x[13], x[14], x[15]],
         [x[4], x[5], x[6],  x[7],  x[16], x[17], x[18], x[19]],
         [x[8], x[9], x[10], x[11], x[20], x[21], x[22], x[23]]]

    x에 대한 reshape 연산(view, t, transpose, permute, contiguous, reshape 등)을
    수행하여 y를 구성해야 합니다.
    입력 tensor는 수정되어서는 안 됩니다.

    Args:
        x: (24,) shape의 tensor

    Returns:
        y: 위에서 설명한 대로 (3, 8) shape의 재구성된 x 버전.
    """
    y = None
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    ##########################################################################
    z = x.view(2, 3, 4)
    z_permute = z.permute(1, 0, 2)
    y = z_permute.contiguous().view(3, -1)

    ##########################################################################
    #                            코드 끝                                      #
    ##########################################################################
    return y


def zero_row_min(x: Tensor) -> Tensor:
    """
    입력 tensor x의 복사본을 return하며, 각 행의 최소값은 0으로 설정됩니다.

    예를 들어, x가 다음과 같을 때:
    x = torch.tensor([
          [10, 20, 30],
          [ 2,  5,  1]])

    y = zero_row_min(x)는 다음과 같아야 합니다:
    torch.tensor([
        [0, 20, 30],
        [2,  5,  0]
    ])

    구현 시 reduction 및 인덱싱 연산을 사용해야 합니다.
    파이썬 반복문 (list comprehension 포함) 을 사용해서는 안 됩니다.
    입력 tensor는 수정되어서는 안 됩니다.

    Args:
        x: (M, N) shape의 tensor

    Returns:
        y: (M, N) shape의 tensor. x의 복사본이지만 각 행의 최소값은 0으로 대체됩니다.
    """
    y = None
    ##########################################################################
    #                      TODO: 이 함수를 구현하세요.                       #
    ##########################################################################
    y = x.clone()
    min_val, min_idx = y.min(dim=1)
    y[torch.arange(y.shape[0]), min_idx] = 0
    ##########################################################################
    #                            코드 끝                                   #
    ##########################################################################
    return y


def batched_matrix_multiply(
    x: Tensor, y: Tensor, use_loop: bool = True
) -> Tensor:
    """
    (B, N, M) shape의 tensor x와 (B, M, P) shape의 tensor y 간의 배치 행렬 곱을 수행합니다.

    use_loop 값에 따라, 이 함수는 batched_matrix_multiply_loop 또는
    batched_matrix_multiply_noloop을 호출하여 실제 계산을 수행합니다.
    여기서는 아무것도 구현할 필요가 없습니다.

    Args:
        x: (B, N, M) shape의 tensor
        y: (B, M, P) shape의 tensor
        use_loop: 명시적인 파이썬 반복문을 사용할지 여부.

    Returns:
        z: (B, N, P) shape의 tensor. 여기서 (N, P) shape의 z[i]는 (N, M) shape의 x[i]와
           (M, P) shape의 y[i] 간의 행렬 곱 결과입니다.
           출력 z는 x와 동일한 dtype을 가져야 합니다.
    """
    if use_loop:
        return batched_matrix_multiply_loop(x, y)
    else:
        return batched_matrix_multiply_noloop(x, y)


def batched_matrix_multiply_loop(x: Tensor, y: Tensor) -> Tensor:
    """
    (B, N, M) shape의 tensor x와 (B, M, P) shape의 tensor y 간의 배치 행렬 곱을 수행합니다.

    이 구현은 출력을 계산하기 위해 배치 차원 B에 대한 반복문을 사용합니다 (단 for loop 1개).

    Args:
        x: (B, N, M) shape의 tensor
        y: (B, M, P) shape의 tensor

    Returns:
        z: (B, N, P) shape의 tensor. 여기서 (N, P) shape의 z[i]는 (N, M) shape의 x[i]와
           (M, P) shape의 y[i] 간의 행렬 곱 결과입니다.
           출력 z는 x와 동일한 dtype을 가져야 합니다.
    """
    z = None
    ###########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    ###########################################################################
    batch_list = []
    for i in range(x.shape[0]):
        batch_list.append(torch.mm(x[i], y[i]))

    z = torch.stack(batch_list)
    ###########################################################################
    #                           코드 끝                                        #
    ###########################################################################
    return z


def batched_matrix_multiply_noloop(x: Tensor, y: Tensor) -> Tensor:
    """
    (B, N, M) shape의 tensor x와 (B, M, P) shape의 tensor y 간의 배치 행렬 곱을 수행합니다.
    이 구현은 파이썬 반복문(list comprehension 포함)을 사용해서는 안 됩니다.

    힌트: torch.bmm

    Args:
        x: (B, N, M) shape의 tensor
        y: (B, M, P) shape의 tensor

    Returns:
        z: (B, N, P) shape의 tensor. 여기서 (N, P) shape의 z[i]는 (N, M) shape의 x[i]와
           (M, P) shape의 y[i] 간의 행렬 곱 결과입니다.
           출력 z는 x와 동일한 dtype을 가져야 합니다.
    """
    z = None
    ###########################################################################
    #                     TODO: 여기에 코드를 작성하세요                            #
    ###########################################################################
    z = torch.bmm(x, y)
    ###########################################################################
    #                            코드 끝                                        #
    ###########################################################################
    return z


def normalize_columns(x: Tensor) -> Tensor:
    """
    행렬 x의 열을 각 열의 평균을 빼고 표준 편차로 나누어 정규화합니다.
    새 tensor를 return해야 합니다. 입력은 수정되어서는 안 됩니다.

    (M, N) shape의 입력 tensor x가 주어졌을 때, y[i, j] = (x[i, j] - mu_j) / sigma_j인
    (M, N) shape의 출력 tensor y를 생성합니다. 여기서 mu_j는 x[:, j] 열의 평균입니다.

    구현 시 파이썬 반복문(list comprehension 포함)을 사용해서는 안 됩니다.
    tensor에 대한 기본 산술 연산(+, -, *, /, **, sqrt), reduction 함수, broadcasting을
    용이하게 하는 reshape 연산만 사용할 수 있습니다.
    torch.mean, torch.std 또는 그 인스턴스 메서드 변형인 x.mean, x.std를 사용해서는 안 됩니다.

    Args:
        x: (M, N) shape의 tensor.

    Returns:
        y: 위에서 설명한 (M, N) shape의 tensor. 입력 x와 동일한 dtype을 가져야 합니다.
    """
    y = None
    ##########################################################################
    #                     TODO: 여기에 코드를 작성하세요                           #
    ##########################################################################
    col_sum = x.sum(dim=0)
    means = col_sum / x.shape[0]

    minus_means_to_x = x - means
    
    stds = (((minus_means_to_x ** 2).sum(dim=0)) / (x.shape[0] - 1)).sqrt()

    y = minus_means_to_x / stds
    ##########################################################################
    #                            코드 끝                                       #
    ##########################################################################
    return y
