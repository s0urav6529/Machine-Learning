
import math
learning_rate = 0.1

def sigmoid(result):
    return 1 / (1 + pow(math.e, -result))

def determine_error(actual, target):
    return ((actual - target) ** 2 ) / 2

def output_l_deltas(l2_output, target):
    return l2_output * (1 - l2_output) * (target - l2_output)

def input_l_deltas(weights, delta, hidden_output):
    input_deltas = []
    for i in range(len(weights)):
        sum = 0
        for j in range(len(weights)):
            sum += weights[j][i]*delta
        input_deltas.append(hidden_output[i]*(1-hidden_output[i])*sum)

    return input_deltas

def hidden_w_update(weights, delta, output):
    for i in range(len(weights)):
        weights[i] += learning_rate * delta * output
    return weights

def input_w_update(weights, deltas, x):
    for i in range(len(weights)):
        for j in range(len(weights)):
            weights[i][j] += learning_rate * deltas[i] * x[j]
    return weights

def weighted_sum(inputs, weights):
    ans = 0
    for input_value, weight in zip(inputs, weights):
        ans += input_value * weight
    return ans

def first_l_output(inputs, weights):
    hidden_outputs = []
    for w in weights:
        hidden_input = weighted_sum(inputs, w)
        hidden_output = sigmoid(hidden_input)
        hidden_outputs.append(hidden_output)

    hidden_outputs.append(1)
    return hidden_outputs

def hidden_l_output(inputs, weights):
    final_input = 0;
    for i in range(len(inputs)):
        final_input += inputs[i]*weights[i]
    final_output = sigmoid(final_input)
    return final_output


input_x = [[0, 0, 1],
           [0, 1, 1],
           [1, 0, 1],
           [1, 1, 1]]

input_w = [[0.17, 0.31, 0.27],
           [0.10, 0.29, 0.33]]

hidden_w = [0.37, 0.26, 0.48]

#Output we want
target = [0, 1, 1, 0]

Overall_error=0

for k in range(4):
    for l in range(1000):

        #input level output
        first_l_op = first_l_output(input_x[k], input_w)

        #final ouput also
        hidden_op = hidden_l_output(first_l_op, hidden_w)

        #Error after every iteration
        Error = determine_error(hidden_op, target[k])

        Output=hidden_op

        #here the backword propagation starts

        #calculating delta of w after every iteration
        delta = output_l_deltas(hidden_op, target[k])

        #new w values after each iteration for output layes
        hidden_w = hidden_w_update(hidden_w, delta, hidden_op)

        #calculating delta of w except output layers
        delta = input_l_deltas(input_w, delta, first_l_op)

        # new w values after each iteration except output layers
        input_w = input_w_update(input_w, delta, input_x[k])

    #Calculating Overall error
    Overall_error =Overall_error + ((target[k]-Output) ** 2)

    #showing Error after Learnig One input
    print(f'Output = {Output} Error = {format(Error*100,".2f")}%')

Overall_error = Overall_error/ (2 * 4)
print(f'Overall Error is = {Overall_error} means {format(Overall_error * 100, ".2f")}%')

