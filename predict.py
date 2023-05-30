import pathlib
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

DATA_DIR = pathlib.Path(".")
MODEL_GAS_1 = pathlib.Path(__file__).parent.joinpath("nn_gas1.pt")
MODEL_GAS_21 = pathlib.Path(__file__).parent.joinpath("nn_gas2_part1.pt")
MODEL_GAS_22 = pathlib.Path(__file__).parent.joinpath("nn_gas2_part2.pt")


class NetGas(nn.Module):
    def __init__(self, input_size=24):
        super(NetGas, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc21 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc21(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self, input_size=24):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисление предсказаний.

    Параметры:
        df:
          датафрейм, содержащий строки из тестового множества.
          Типы и имена колонок совпадают с типами и именами в ноутбуке, не содержит `np.nan` или `np.inf`.

    Результат:
        Датафрейм предсказаний.
        Должен содержать то же количество строк и в том же порядке, а также колонки `target0` и `target1`.
    """
    df_predict = df.copy(deep=True)

    # df_predict['feature13_exp'] = np.log(df_predict['feature13'])

    df_gas1 = df_predict[df_predict['feature4'] == 'gas1']
    df_gas2_part1 = df_predict[(df_predict['feature4'] == 'gas2') & (df_predict['feature8'] >= -35)]
    df_gas2_part2 = df_predict[(df_predict['feature4'] == 'gas2') & (df_predict['feature8'] < -35)]

    df_predict['target0'] = 45
    df_predict['target1'] = 22



    features2_part1 = ['feature0',
 'feature1',
 'feature2',
 'feature3',
 'feature5',
 'feature6',
 'feature7',
 'feature8',
 'feature9',
 'feature10',
 'feature11',
 'feature12',
 'feature13',
 'feature14',
 'feature15',
 'feature16',
 'feature17',
 'feature18',
 'feature20',
 'feature22']

    features2_part2 = ['feature0',
                'feature1',
                'feature2',
                'feature3',
                'feature5',
                'feature6',
                'feature7',
                'feature9',
                'feature10',
                'feature11',
                'feature12',
                'feature13',
                'feature14',
                'feature15',
                'feature16',
                'feature17',
                'feature18',
                'feature20',
                'feature22']

    features1 = ['feature0',
     'feature2',
     'feature3',
     'feature6',
     'feature9',
     'feature10',
     'feature11',
     'feature12',
     'feature13',
     'feature14',
     'feature15',
     'feature16',
     'feature18',
     'feature19',
     'feature20',
     'feature21']

    model = NetGas(input_size=len(features1))
    model.load_state_dict(torch.load(MODEL_GAS_1))
    model.eval()

    model2_part1 = NetGas(input_size=len(features2_part1))
    model2_part1.load_state_dict(torch.load(MODEL_GAS_21))
    model2_part1.eval()

    model2_part2 = Net2(input_size=len(features2_part2))
    model2_part2.load_state_dict(torch.load(MODEL_GAS_22))
    model2_part2.eval()

    x_test = df_gas1.loc[:, features1]
    x_test21 = df_gas2_part1.loc[:, features2_part1]
    x_test22 = df_gas2_part2.loc[:, features2_part2]

    x_test_var = Variable(torch.FloatTensor(x_test.values), requires_grad=False)
    x_test_var21 = Variable(torch.FloatTensor(x_test21.values), requires_grad=False)
    x_test_var22 = Variable(torch.FloatTensor(x_test22.values), requires_grad=False)

    with torch.no_grad():
        y_predict = model(x_test_var)
        y_predict21 = model2_part1(x_test_var21)
        y_predict22 = model2_part2(x_test_var22)

    df_predict = pd.DataFrame(index=df.index)

    df_predict.loc[df_gas1.index, ['target0', 'target1']] = y_predict.numpy()
    df_predict.loc[df_gas2_part1.index, ['target0', 'target1']] = y_predict21.numpy()
    df_predict.loc[df_gas2_part2.index, ['target0', 'target1']] = y_predict22.numpy()

    return df_predict
