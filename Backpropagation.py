def sigmoid(x):
  import math
  y =  1/ (1 + math.e**(-x))
  return y


#Our input layers (Xi):
x1 = [0,0,1,1]
x2 = [0,1,0,1]
#Our bias for hidden layers (Hj):
b1 = 1
b2 = 1
#Our hidden layer (Hj):
H1 = [0,0,0,0]
H2 = [0,0,0,0]
#Our output layers (Yk) and bias:
y = [0,0,0,0]
b3 = 1
#Our target Output (Tn):
t = [0.017622,0.981504,0.981491,0.022782]   #[0,1,1,0] // [0.017622,0.981504,0.981491,0.022782] 
#Our weights:
Wb1 = 0.862518
Wb2 = 0.834986
Wbb1 = 0.036498
Wij = [-0.155797,-0.505997,0.282885,-0.864449]
Wjk = [-0.430437,0.481210]
#Wij = [-0.155797,-0.505997,0.282885,-0.864449] #[W11,W12,W21,W22] for x1 and x2 to hidden layers 
#Wjk = [-0.430437,0.481210] #[W11,W21] for hidden layers to result k
#Our learning rate (n):
n = 0.5
#Our error ratess:
error_n = 0
error_k = [0,0,0,0]
error_j = [0,0,0,0]
it = 0
while True:
  #Lets calculate our hidden layers (Hj) and output layer:
  it +=1
  import math
  for i in range(len(x1)):
    H1[i] = sigmoid(Wij[0]*x1[i] + Wij[2]*x2[i] +Wb1) # we are using the sigmoid function
    H2[i] = sigmoid(Wij[1]*x1[i] + Wij[3]*x2[i] +Wb2)
  #Now, we can calculate our output layer (Yn)
  for i in range(len(x1)): 
    y[i] = sigmoid(Wjk[0] * H1[i] + Wjk[1] * H2[i]+  +Wbb1)
  #Then, we would calculate the error delta(n), delta(k) and delta(j):
  for i in range(len(x1)):
    error_k[i] = y[i] *( 1- y[i]) * (t[i] - y[i])
    
  error_n = sum(error_k)

  error_j_h1 = [0,0,0,0]
  error_j_h2 = [0,0,0,0]
  for i in range(len(x1)): 
    error_j_h1[i] = H1[i] * (1 - H1[i]) * Wjk[0] * error_k[i]
    error_j_h2[i] = H2[i] * (1 - H2[i]) * Wjk[1] * error_k[i]

  if abs(error_n) < 0.0000000001: #]abs(error_n) < 0.000001
    print()
    print("The Yn are: ",y)
    print("Wb1: " ,Wb1)
    print("Wb2: " ,Wb2)
    print("Wbb1: " ,Wbb1)
    print("Wij: ",Wij)
    print("Wjk: ",Wjk)
    print("iteraciones:", it)
    break
  #Now, we would update the weights:
  #Our weights: Wij(4),  Wjk(2), Wb1, Wb2, Wbb1
  #1. Update Wb1,Wb2,Wbb1:
  Wb1 = Wb1 + n *b1 * sum(error_j_h1)
  Wb2 = Wb2 + n *b1 * sum(error_j_h2)
  Wbb1 = Wbb1 + n *b1 * sum(error_k)


  #2. Update Wjk
  Var_Wjk_1 = [0,0,0,0]
  Var_Wjk_2 = [0,0,0,0]
  for i in range(len(x1)): 
    Var_Wjk_1[i] = n * error_k[i]* H1[i]#[W11,W21] 
    Var_Wjk_2[i] = n * error_k[i]* H2[i]#[W11,W21] 

  Var_Wjk1 = sum(Var_Wjk_1)
  Var_Wjk2 = sum(Var_Wjk_2)

  Var_Wjk = [Var_Wjk1,Var_Wjk2]

  for i in range(len(Wjk)): 
    Wjk[i] =  Wjk[i] + Var_Wjk[i]  #[W11,W21]
  # Update Wij:

  Var_Wij_0 = [0,0,0,0]
  Var_Wij_2 = [0,0,0,0]
  Var_Wij_1 = [0,0,0,0]
  Var_Wij_3 = [0,0,0,0]
  for i in range(len(x1)):
    Var_Wij_0[i] = n*error_j_h1[i] * x1[i]
    Var_Wij_2[i] = n*error_j_h1[i] * x2[i]
    Var_Wij_1[i] = n*error_j_h2[i] * x1[i]
    Var_Wij_3[i] = n*error_j_h2[i] * x2[i]

  Var_Wij0 = sum(Var_Wij_0)
  Var_Wij1 = sum(Var_Wij_1)
  Var_Wij2 = sum(Var_Wij_2)
  Var_Wij3 = sum(Var_Wij_3)

  Wij[0] = Wij[0] + Var_Wij0
  Wij[2] = Wij[2] + Var_Wij2
  Wij[1] = Wij[1] + Var_Wij1
  Wij[3] = Wij[3] + Var_Wij3
  
