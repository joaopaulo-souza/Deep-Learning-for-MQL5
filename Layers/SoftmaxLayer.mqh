//+------------------------------------------------------------------+
//|   Softmax Layer                                                  |
//+------------------------------------------------------------------+

class softmaxLayer : public DeepLearning
  {
public:
   virtual matrix Output(matrix &X);
   virtual matrix GradDescent(matrix &Ey);
private:
   matrix M; 
  };

//+------------------------------------------------------------------+
//|     Softmax                                                      |
//+------------------------------------------------------------------+
matrix softmaxLayer::Output(matrix &X)
{
matrix Y;
Y.Init(X.Rows(),X.Cols());
double Sum;
Sum = 0;

for(int i=0;i<X.Rows();i++)
  {Sum = Sum + MathExp(X[i][0]);}
for(int i=0;i<Y.Rows();i++)
  {Y[i][0] = MathExp(X[i][0])/Sum;}

M.Init(Y.Rows(),Y.Rows());

for(int j=0;j<M.Cols();j++)
  {for(int i=0;i<M.Rows();i++)
     {M[i][j] = Y[i][0];}}

return Y; 
}
matrix softmaxLayer::GradDescent(matrix &Ey)
{
matrix S;
matrix I;
I.Init(M.Rows(),M.Cols());
I.Identity();
S = M * (I - M.Transpose());
S = S.MatMul(Ey);
return S;
}
