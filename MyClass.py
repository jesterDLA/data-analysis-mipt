import numpy as np
import torch as T
device = 'cpu'

class MyLogisticRegression:

    def __init__(self, number_features) -> None:
        self.number_features = number_features
    
    
    def forward(self, x: T.tensor,
            w: T.tensor,
            b: T.tensor):
        z = T.dot(x, w).reshape(1)
        z += b
        p = 1 / (1 + T.exp(-z))
        return p

    def initialize_b_and_w(self):
        lo = -0.01; hi = 0.01
        self.w = T.rand((self.number_features), dtype=T.float32, requires_grad=True).to(device)
        self.w = (hi - lo) * self.w + lo
        self.w.grad = T.zeros(self.number_features)
        self.w.retain_grad()
        
        self.b = T.zeros((1), dtype=T.float32, requires_grad=True).to(device)
        self.b.grad = T.zeros(1)
        self.b.retain_grad()

    def fit(self, times, train_x, train_y, lrn_rate = 0.0001):
        self.initialize_b_and_w()
        len_data = len(train_x)
        indices = np.arange(len_data)

        for epoch in range(0, times):
            tot_loss = 0
            tot_loss = T.zeros((1), dtype=T.float32, requires_grad=True).to(device)
            tot_loss.grad = T.zeros(1)
            tot_loss.retain_grad()
            
            np.random.shuffle(indices)  
            for ii in range(len(indices)):
                i = indices[ii]
                x = train_x[i]
                target = train_y[i]
                oupt = self.forward(x, self.w, self.b)
                loss = (oupt - target).pow(2).sum()
                tot_loss = loss + tot_loss
                
        #     tot_loss = tot_loss + T.norm(w, p=2) # l2 reg
        #     tot_loss = tot_loss + T.norm(w, p=1) # l1 reg

            tot_loss.backward(retain_graph=True)  # compute gradients

            self.w.data += -1 * lrn_rate * self.w.grad.data
            self.b.data += -1 * lrn_rate * self.b.grad.data

            self.w.grad = T.zeros(self.number_features)
            self.b.grad = T.zeros(1)

            if epoch % 4 == 0:
                print("epoch = %4d " % epoch, end="")
                print("   loss = %6.4f" % (tot_loss / len_data))

        return self.w, self.b

    def predictions(self, data_for_predict):
        prediction = np.empty(len(data_for_predict), dtype=np.float32)

        for i, pred_i in enumerate(data_for_predict):
            # print(type(pred_i), type(self.w), type(self.b))
            prediction[i] = self.forward(pred_i, self.w, self.b)

        return prediction


# a = np.array([[56, 78], [78, 56]])
# print(len(a))