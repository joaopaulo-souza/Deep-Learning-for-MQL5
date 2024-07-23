//+------------------------------------------------------------------+
//|   Activation Layer                                               |
//+------------------------------------------------------------------+

class ActivationLayer : public DeepLearning
  {
public:
   
   void   InitLayer(ActFunction af);
   virtual matrix Output(matrix &X);
   virtual matrix GradDescent(matrix &Ey);
   
private :
   
   //Activation function
   ActFunction AF;
   //Input that needs to be saved for gradient descent
   matrix Xs;  
  };

//+------------------------------------------------------------------+
//|    Activation Layer                                              |
//+------------------------------------------------------------------+

void ActivationLayer::InitLayer(ActFunction af)
{
AF = af;
}
matrix ActivationLayer::Output(matrix &X)
{
matrix Y;

Xs = X;
if(AF == SIGMOID) Y = Sig(X);
if(AF == TANH)    Y = Tanh(X);
if(AF == RELU)    Y = ReLU(X);

return Y;
}
matrix ActivationLayer::GradDescent(matrix &Ey)
{
matrix dPhi;

matrix Ex;
if(AF == SIGMOID) dPhi = dSig(Xs);
if(AF == TANH)    dPhi = dTanh(Xs);
if(AF == RELU)    dPhi = dReLU(Xs);

Ex = Ey * dPhi;
return Ex;
}
