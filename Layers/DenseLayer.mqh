//+------------------------------------------------------------------+
//|   Dense Layer                                                    |
//+------------------------------------------------------------------+

class DenseLayer : public DeepLearning
  {
public:
   //Initializes the layer weights and bias from the number of outputs
   //LR is Learning rate and Op is the optimization method
   void InitLayer(int N_entries, int N_outputs, double LR, Optim Op = STD);
   //Calculates the output of the layer from an input
   virtual matrix Output(matrix &X);
   //Propagates the error  
   virtual matrix GradDescent(matrix &Ey);
   //Updates the weights
   virtual void   Update(void);
   //Save the weights, k is the index the layer;
   virtual void   SaveWeights(int k,string IAname);
   //Load the weights
   virtual void   LoadWeights(int k,string IAname);
   
   //Configurar parâmetros do ADAM
   virtual void   SetAdam(double B1, double B2, double Alph);
private:
   //Pesos da camada
   matrix W;
   //Grad dos Pesos
   matrix dW;
   //Bias da camada
   matrix B;
   //Grad do Bias
   matrix dB;
   //Vetor de entrada 
   matrix Xe;
   //Taxa de aprendizagem
   double N;
   //Método de otimzação
   Optim OP;
   
   //ADAM 
   //===============
   //Contador de iterações
   ulong it; 
   //m
   matrix mW, mB;
   //v
   matrix vW, vB;
   //hiper parâmetros
   double beta1, beta2, alpha; 
      
  };

//+------------------------------------------------------------------+
//|   DenseLayer                                                     |
//+------------------------------------------------------------------+
void DenseLayer::InitLayer(int N_entries, int N_outputs, double LR, Optim Op = STD)
{
   W.Init(N_outputs,N_entries);
   B.Init(N_outputs,1);
   N = LR;
   OP = Op;
   W = InitWeights(W);
   B = InitWeights(B);
   
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
matrix DenseLayer::Output(matrix &X)
{
matrix Y;
Xe = X;
Y = W.MatMul(X) + B;
return Y;
}
matrix DenseLayer::GradDescent(matrix &Ey)
{
   matrix Ex; 
   Ex = W.Transpose().MatMul(Ey);
   
   dW = Ey.MatMul(Xe.Transpose());
   dB = Ey;
   return Ex;
}
void DenseLayer::Update(void)
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
void DenseLayer::SaveWeights(int k, string IAname)
{

   string csv_name;
   csv_name = IAname + "\WLayer" + IntegerToString(k);
   SaveMatrix(W,csv_name);
   csv_name = IAname + "\BLayer" + IntegerToString(k);
   SaveMatrix(B,csv_name);

} 

void DenseLayer::LoadWeights(int k, string IAname)
{

   string csv_name;
   csv_name = IAname + "\WLayer" + IntegerToString(k);
   W = LoadMatrix(csv_name);
   csv_name = IAname + "\BLayer" + IntegerToString(k);
   B = LoadMatrix(csv_name);

}
void DenseLayer::SetAdam(double B1,double B2,double Alph)
{
beta1 = B1;
beta2 = B2;
alpha = Alph;
}