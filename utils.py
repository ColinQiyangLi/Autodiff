import numpy as np

def topo_sort(op):
    '''
    A generator that yields the variable with the topological order in the computational graph.
    :param op: The initial node
    :return:
    '''
    yield op
    if op.is_variable:
        return
    for key in op.ops:
        pre = op.ops[key]
        pre.out_degree -= 1
        if pre.out_degree == 0:
            for item in topo_sort(pre):
                yield item

def broadcast_gradient(grad, shape):
    '''
    Broadcast a gradient to a shape with more dimension (broadcast 1 to non-1 size)
    :param grad:
    :param shape:
    :return:
    '''
    assert len(grad.shape) == len(shape)
    n_dim = len(grad.shape)
    dims = []
    for i in range(n_dim):
        if shape[i] == 1:
            dims.append(i)
    return np.sum(grad, axis = tuple(dims), keepdims=True)

class operator():
    def __init__(self):
        self.out_degree = 0
        self.is_variable = False

    def forward(self, **kwargs):
        self.ops = kwargs
        for key in kwargs:
            self.ops[key].out_degree += 1
        self.value = self.compute_forward(**self.ops)
        self.grad = np.zeros_like(self.value, dtype = np.float32)

    def compute_forward(self, **kwargs):
        raise NotImplementedError

    def backward(self):
        self.grad = 1.0 # assume that we always backprop from a scalar
        for op in topo_sort(self):
            if not op.is_variable:
                op.compute_derivatives(op.grad, **op.ops)

    def compute_derivatives(self, **kwargs):
        raise NotImplementedError

class variable(operator):
    def __init__(self, value):
        super(variable, self).__init__()
        self.value = value
        self.is_variable = True
        self.grad = np.zeros_like(self.value, dtype = np.float32)

class matmul(operator):
    def __init__(self, x, y):
        super(matmul, self).__init__()
        input = {"x": x, "y": y}
        self.forward(**input)

    def compute_forward(self, x, y):
        return np.matmul(x.value, y.value)

    def compute_derivatives(self, grad, x, y):
        x.grad += np.matmul(grad, y.value.T)
        y.grad += np.matmul(x.value.T, grad)

class add(operator):
    def __init__(self, x, y):
        super(add, self).__init__()
        input = {"x": x, "y": y}
        self.forward(**input)

    def compute_forward(self, x, y):
        assert len(x.value.shape) == len(y.value.shape)
        return np.add(x.value, y.value)

    def compute_derivatives(self, grad, x, y):
        x.grad += broadcast_gradient(grad, x.value.shape)
        y.grad += broadcast_gradient(grad, y.value.shape)

class softmax(operator):
    def __init__(self, x):
        super(softmax, self).__init__()
        input = {"x": x}
        self.forward(**input)

    def compute_forward(self, x):
        assert len(x.value.shape) == 2
        e_x = np.exp(x.value - np.max(x.value, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def compute_derivatives(self, grad, x):
        y = self.value
        A = -y[:, np.newaxis, :] * y[:, :, np.newaxis]
        B = np.multiply(y[:, np.newaxis, :], np.eye(y.shape[1])[np.newaxis, :, :])
        J = A + B
        x.grad = np.matmul(grad[:, np.newaxis, :], J).reshape(x.value.shape)

class neg_loglikelihood(operator):
    def __init__(self, x, y):
        super(neg_loglikelihood, self).__init__()
        input = {"x": x, "y": y}
        self.forward(**input)

    def compute_forward(self, x, y):
        assert x.value.shape == y.value.shape
        assert len(x.value.shape) == 2
        return -np.sum(np.log(x.value) * y.value)

    def compute_derivatives(self, grad, x, y):
        x.grad -= grad * y.value / x.value
        y.grad -= grad * np.log(x.value)

class mean(operator):
    def __init__(self, x, axis = -1):
        super(mean, self).__init__()
        input = {"x": x}
        self.axis = axis
        self.forward(**input)

    def compute_forward(self, x):
        return np.mean(x.value, axis = self.axis)

    def compute_derivatives(self, grad, x):
        x.grad += np.expand_dims(grad, axis = self.axis) * np.ones_like(x.value) / x.value.shape[self.axis]

class sum(operator):
    def __init__(self, x, axis = -1):
        super(sum, self).__init__()
        input = {"x": x}
        self.axis = axis
        self.forward(**input)

    def compute_forward(self, x):
        return np.sum(x.value, axis = self.axis)

    def compute_derivatives(self, grad, x):
        x.grad += np.expand_dims(grad, axis = self.axis) * np.ones_like(x.value)

class multiply(operator):
    def __init__(self, x, y):
        super(multiply, self).__init__()
        input = {"x": x, "y": y}
        self.forward(**input)

    def compute_forward(self, x, y):
        return np.multiply(x.value, y.value)

    def compute_derivatives(self, grad, x, y):
        x.grad += y.value
        y.grad += x.value

class optimizer():
    def __init__(self, vars, hype):
        self.vars = vars
        self.lr = hype["learning_rate"]

    def optimize(self):
        for var in self.vars:
            var.value -= var.grad * self.lr

    def clear_grad(self):
        for var in self.vars:
            var.grad = np.zeros_like(var.value, dtype = np.float64)

class momentum_optimizer(optimizer):
    def __init__(self, vars, hype):
        super(momentum_optimizer, self).__init__(vars, hype)
        self.beta = hype["beta"]  # momentum ratio
        self.pre_update = {}

    def optimize(self):
        for var in self.vars:
            if not var in self.pre_update:
                pre_update = np.zeros_like(var.value)
            else:
                pre_update = self.pre_update[var]
            update = -var.grad * self.lr + pre_update * self.beta
            var.value += update
            self.pre_update[var] = update
