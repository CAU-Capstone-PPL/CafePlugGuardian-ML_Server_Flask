import torch
import torch.nn as nn
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import os



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#모델선언
class GRUModel(nn.Module):
   def __init__(self, input_dim, hidden_dim, layer_dim, num_classes):
      super(GRUModel, self).__init__()

      # Defining the number of layers and the nodes in each layer
      self.layer_dim = layer_dim
      self.hidden_dim = hidden_dim
      self.name = "GRU"
      # GRU layers
      self.gru = nn.GRU(
         input_dim, hidden_dim, layer_dim, batch_first=True
      )

      # Fully connected layer
      self.fc = nn.Linear(hidden_dim, num_classes)

   def forward(self, x):
      # Initializing hidden state for first input with zeros
      h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

      # Forward propagation by passing in the input and hidden state into the model
      out, _ = self.gru(x, h0.detach())

      # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
      # so that it can fit into the fully connected layer
      out = out[:, -1, :]

      # Convert the final state to our desired output shape (batch_size, output_dim)
      out = self.fc(out)

      return out
#모델 끝
app = Flask(__name__)
CORS(app)

@app.route('/predict2', methods=['POST'])
def predict2():
   try:
      if request.method=='POST':
      # 전달된 JSON 데이터 가져오기
         input_json = request.json

      # JSON 데이터를 활용하는 로직
         print('Received JSON data:')
         model_path=os.path.join('CafePlugGuardian_ML_Model.pt')
         model = torch.load(model_path,map_location=torch.device('cpu'))
         model.eval()
      # 예시: JSON 데이터를 딕셔너리로 활용


         result_count = np.array([0, 0])
         x_input = np.array(input_json.get('current', []))
         print(x_input)# 예시: 'filter_current' 키에 해당하는 값을 가져옴
         x_normalized = (x_input - min(x_input)) / (max(x_input) - min(x_input))



         for i in range(500 - 33 + 1):
            x_test = torch.Tensor(x_normalized[i: i + 33]).unsqueeze(0).unsqueeze(1).to(DEVICE)
            result = np.argmax(model(x_test).cpu().detach().numpy(), axis=1)

            for fin in result:
               result_count[fin] += 1

         print('Prediction result:', result_count)

         if result_count[1] >= 380:
            print("isPermited:True")
            return jsonify({'isPermitted':'True'})
         elif result_count[0] >= 380:
            print("isPermited:False")
            return jsonify({'isPermitted':'False'})
         else :
            print("isPermited:New Item")
            return jsonify({'isPermitted':'New Item'})

   except Exception as e:
      # 오류 메시지 출력
      print(f'Error processing request: {str(e)}')
      return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
   app.run('0.0.0.0', port=5000, debug=True)

