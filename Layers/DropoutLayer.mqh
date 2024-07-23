//+------------------------------------------------------------------+
//|   Dropout Layer                                                  |
//+------------------------------------------------------------------+

class DropoutLayer : public DeepLearning
  {
public:

   //Initializes the layer weights and bias from the number of neurons
   void InitLayer(int N_entries, int N_outputs, double LR, double Drop,Optim Op = STD);
   //Calculates the output of the litter from an input
   virtual matrix Output(matrix &X);
   //Propagates the error
   virtual matrix GradDescent(matrix &Ey);
   //Updates the weights
   virtual void   Update(void);
   //Save the weights, k is the index of the layer;
   virtual void   SaveWeights(int k,string IAname);
   //Load the weights 
   virtual void   LoadWeights(int k,string IAname);
   //Set a different value for the DropOut rate
   virtual void   SetDrop(double Drop);
   
   //ADAM
   virtual void   SetAdam(double B1,double B2,double Alph);
   
private:
   //weight sof the layer
   matrix W;
   //Gradients of the weights 
   matrix dW;
   //Bias of the layer
   matrix B;
   //Bias gradient
   matrix dB;
   //Input vector
   matrix Xe;
   //Learning rate
   double N;
   //dropout rate
   double drop;
   //Dropout matrix
   matrix D;
   //Optimization method
   Optim OP;
   
      //ADAM 
   //===============
   //Iterations counter
   ulong it; 
   //m
   matrix mW, mB;
   //v
   matrix vW, vB;
   //hiperparameters 
   double beta1, beta2, alpha; 
  };

//+------------------------------------------------------------------+
//|   DropoutLayer                                                               |
//+------------------------------------------------------------------+
void DropoutLayer::InitLayer(int N_entries, int N_outputs, double LR, double Drop,Optim Op = STD)
{
   W.Init(N_outputs,N_entries);
   B.Init(N_outputs,1);
   N = LR;
   OP = Op;
   W = InitWeights(W);
   B = InitWeights(B);

   drop = Drop;

   if(OP == ADAM)
     {
      it = 0;
      beta1 = 0.9;
      beta2 = 0.999;
      alpha = N;
      
      mW.Init(W.Rows(),W.Cols());
      vW.Init(W.Rows(),W.Cols());  
      mW = ZeroMatrix(mW);
      vW = ZeroMatrix(vW);
      
      mB.Init(B.Rows(),B.Cols());
      vB.Init(B.Rows(),B.Cols());
      mB = ZeroMatrix(mB);
      vB = ZeroMatrix(vB);
     }
}
matrix DropoutLayer::Output(matrix &X)
{

matrix Y;
Xe = X;
Y = W.MatMul(X) + B;

//Construção da matriz de Dropout
matrix Dp;
Dp.Init(Y.Rows(),Y.Cols());
for(int i=0;i<Dp.Rows();i++)
  {for(int j=0;j<Dp.Cols();j++)
     {if(rand()/32780.0 < drop)  Dp[i][j] = 1;
      else Dp[i][j] = 0; }}
D = Dp;

Y = Y * Dp;
return Y;
}
matrix DropoutLayer::GradDescent(matrix &Ey)
{
   matrix Ex;
   Ey = Ey * D;
   Ex = W.Transpose().MatMul(Ey);

   
   dW = Ey.MatMul(Xe.Transpose());
   dB = Ey;
   
   return Ex;
}
void DropoutLayer::Update(void)
{
   if(OP == STD)
     {
      W = W - dW*N;
      B = B - dB*N;}
   
   if(OP == ADAM)
     {

      it +=1;
      
      mW = AdamM(mW,dW,beta1);
      vW = AdamV(vW,dW,beta2);
      W = W - Adam(it,mW,vW,beta1,beta2,alpha);
      
      mB = AdamM(mB,dB,beta1);
      vB = AdamV(vB,dB,beta2);
      B = B -Adam(it,mB,vB,beta1,beta2,alpha);
      
     }
}
void DropoutLayer::SaveWeights(int k,string IAname)
{
   string csv_name;
   csv_name = IAname + "\WLayer" + IntegerToString(k);
   SaveMatrix(W,csv_name);
   csv_name = IAname + "\BLayer" + IntegerToString(k);
   SaveMatrix(B,csv_name);
   
} 

void DropoutLayer::LoadWeights(int k,string IAname)
{
   string csv_name;
   csv_name = IAname + "\WLayer" + IntegerToString(k);
   W = LoadMatrix(csv_name);
   csv_name = IAname + "\BLayer" + IntegerToString(k);
   B = LoadMatrix(csv_name);
}
void DropoutLayer::SetDrop(double Drop)
{
drop = Drop;
}
void DropoutLayer::SetAdam(double B1,double B2,double Alph)
{
beta1 = B1;
beta2 = B2;
alpha = Alph;
}