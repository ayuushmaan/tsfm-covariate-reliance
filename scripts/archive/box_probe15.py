"""predict_to_numpy source + Moirai2Module.forward source."""
import inspect

from gluonts.torch.util import predict_to_numpy
print(inspect.getsource(predict_to_numpy))
from uni2ts.model.moirai2 import Moirai2Module
print(inspect.getsource(Moirai2Module.forward))
