 //+------------------------------------------------------------------+
 //|  Loss                                                            |
 //+------------------------------------------------------------------+
  
class Loss : public DeepLearning
{
public: 
   //Defines Cross Entropy for calculating error in classification problems
   double CrossEntropy(matrix &R, matrix &Y);
   //Defines the mean squared error for classification problems; 
   double MeanSquaredError(matrix &R, matrix &Y);
   
   //Defines the derivative of the cross entropy function. (Works with softmax previous activation layer)
   matrix Grad_CE(matrix &R, matrix &Y);
   //Two-class case (Only works with a sigmoid anterior activation layer)
   matrix Grad_BinaryCE(matrix &R, matrix &Y);
   //Defines the derivative of the Quadratic error function.
   matrix Grad_MSE(matrix &R, matrix &Y);
   
   //notas: 
   //R is the expected output
   //Y is the output calculated by the neural network
};

//+------------------------------------------------------------------+
//|   Loss                                                           |
//+------------------------------------------------------------------+
double Loss::CrossEntropy(matrix &R, matrix &Y)
{
double l; 
matrix L;

L = R * MathLog(Y);
l = 0;
for(int i=0;i<L.Rows();i++)
  {for(int j=0;j<L.Cols();j++)
     {   l = l + L[i][j];}};
return -l;
}
double Loss::MeanSquaredError(matrix &R, matrix &Y)
{
double l; 
matrix L;
double c;
L = MathPow(R-Y,2);
c = R.Rows();
l = 0;
for(int i=0;i<L.Rows();i++)
  {for(int j=0;j<L.Cols();j++)
     {   l = l + L[i][j];}}

l = l/c;
return l; 
}     
matrix Loss::Grad_CE(matrix &R, matrix &Y)
{
matrix a;
a = (-1)*(R/Y);
return a;
}
matrix Loss::Grad_BinaryCE(matrix &R,matrix &Y)
{
matrix a;
double c;
c = R.Rows();
a = ((1 - R) / (1 - Y) - R / Y) / c;
return a;
}
matrix Loss::Grad_MSE(matrix &R,matrix &Y)
{
matrix a;
double c; 
c = R.Rows();
a = 2*(Y - R)/c;
return a;
}