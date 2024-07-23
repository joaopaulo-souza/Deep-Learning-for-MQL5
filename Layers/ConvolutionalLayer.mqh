//+------------------------------------------------------------------+
//|   Convolutional Layer                                                               |
//+------------------------------------------------------------------+
class ConvolutionalLayer : public DeepLearning
  {
public:
   //Initialize kernels and layer biases
   //1<= N_outputs <= N_entries; N_entries-N_outputs+1 =KernelSize.  
   void InitLayer(int N_steps, int N_entries, int N_outputs, double LR, CONV_DIR direction = HORZ, Optim Op = STD);
   //Calculates the output of the litter from an input
   virtual matrix Output(matrix &X);
   //Propagate the Error
   virtual matrix GradDescent(matrix &Ey);
   //Update the weights
   virtual void   Update(void);
   //Save the weights, k is the layer index;
   virtual void   SaveWeights(int k,string IAname);
   //Load Weights
   virtual void   LoadWeights(int k,string IAname);
   
   //ADAM
   virtual void   SetAdam(double B1,double B2,double Alph);
   
   
   
private:
   //Kernel layer
   matrix K;
   //Kernel Gradient
   matrix dK;
   //Bias of the layer
   matrix B;
   //Bias gradient
   matrix dB;
   //Input layer
   matrix Xe;
   //Learning rate
   double N;
   //Optimization method
   Optim OP;
   //Convolution direction
   CONV_DIR DIR;
   
   
   //ADAM 
   //===============
   //iteration counter
   ulong it; 
   //m
   matrix mK, mB;
   //v
   matrix vK, vB;
   //hiperparameters 
   double beta1, beta2, alpha; 
  };

//+------------------------------------------------------------------+
//|   Convolutional Layer                                            |
//+------------------------------------------------------------------+

void ConvolutionalLayer::InitLayer(int N_steps, int N_entries, int N_outputs, double LR, CONV_DIR direction = HORZ, Optim Op = STD)
{
   DIR = direction;
   
   if(DIR == VERT)
   { K.Init(N_steps-N_outputs+1,N_entries);
   B.Init(N_steps-K.Rows()+1,N_entries);}
   
   if(DIR == HORZ)
   { K.Init(N_entries,N_steps-N_outputs + 1);
   B.Init(N_entries,N_steps - K.Cols() +1);}
   
   N = LR;
   OP = Op;

   K = InitWeights(K);
   B = InitWeights(B);

   if(OP == ADAM)
     {
      it = 0;
      beta1 = 0.9;
      beta2 = 0.999;
      alpha = N;
      
      mK.Init(K.Rows(),K.Cols());
      vK.Init(K.Rows(),K.Cols());  
      mK = ZeroMatrix(mK);
      vK = ZeroMatrix(vK);
      
      mB.Init(B.Rows(),B.Cols());
      vB.Init(B.Rows(),B.Cols());
      mB = ZeroMatrix(mB);
      vB = ZeroMatrix(vB);
     }

}
matrix ConvolutionalLayer::Output(matrix &X)
{
matrix Y;
Xe = X;
if(DIR == VERT)Y = VertConvV(X,K) + B;
if(DIR == HORZ)Y = HorConvV(X,K) + B;
return Y;
}
matrix ConvolutionalLayer::GradDescent(matrix &Ey)
{
   matrix Ex,Ki,Xi;
   if(DIR == VERT)
   {
   Ki = VertInv(K);
   Ex = VertConvF(Ki,Ey);
   
   Xi = VertInv(Xe);
   dK = VertConvV(Xi,Ey);
   dB = Ey;}
   
   if(DIR == HORZ)
   {
   Ki = HorInv(K);
   Ex = HorConvF(Ki,Ey);
   
   Xi = HorInv(Xe);
   dK = HorConvV(Xi,Ey);
   dB = Ey;}
   return Ex;
}
void ConvolutionalLayer::Update(void)
{
   if(OP == STD)
     {
      K = K - dK*N;
      B = B - dB*N;}
   if(OP == ADAM)
     {

      it +=1;
      
      mK = AdamM(mK,dK,beta1);
      vK = AdamV(vK,dK,beta2);
      K = K - Adam(it,mK,vK,beta1,beta2,alpha);
      
      mB = AdamM(mB,dB,beta1);
      vB = AdamV(vB,dB,beta2);
      B = B - Adam(it,mB,vB,beta1,beta2,alpha);
      
     }
}
void ConvolutionalLayer::SaveWeights(int k,string IAname)
{
   string csv_name;
   csv_name = IAname + "\KLayer" + IntegerToString(k);
   SaveMatrix(K,csv_name);
   csv_name = IAname + "\BLayer" + IntegerToString(k);
   SaveMatrix(B,csv_name);
   
} 

void ConvolutionalLayer::LoadWeights(int k,string IAname)
{
   string csv_name;
   csv_name = IAname+ "\KLayer" + IntegerToString(k);
   K = LoadMatrix(csv_name);
   csv_name = IAname+ "\BLayer" + IntegerToString(k);
   B = LoadMatrix(csv_name);

}
void ConvolutionalLayer::SetAdam(double B1,double B2,double Alph)
{
beta1 = B1;
beta2 = B2;
alpha = Alph;
}
