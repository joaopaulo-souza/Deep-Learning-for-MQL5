//+------------------------------------------------------------------+
//|   Metrics                                                        |
//+------------------------------------------------------------------+

class Metrics : public DeepLearning
  {
public:
  //Used for multiclass problems, transforms the output matrix 
  //from the softmax layer into an output of zeros and ones where 1 
  // is the most likely class
   matrix ArgMax(matrix &X);
   
   //Accuracy = Total number of hits over the total number of samples
   double Accuracy(matrix &R, matrix &Out);
   
   //Mean absolute percentage error 
   double MAPE(matrix &R, matrix &Y);
  };

//+------------------------------------------------------------------+
//|    Metrics                                                       |
//+------------------------------------------------------------------+
matrix Metrics::ArgMax(matrix &X)
{
vector V;
V.Init(X.Cols());

ulong count;
matrix M;
M.Init(X.Rows(),X.Cols());
for(int i=0;i<X.Rows();i++)
   {for(int j=0;j<X.Cols();j++)
      {V[j] = X[i][j];}
   count = V.ArgMax();
   
   for(int j=0;j<X.Cols();j++)
     {if(j == count) M[i][j] = 1;
      if(j != count) M[i][j] = 0;}
    }
return M; 
}

double Metrics::Accuracy(matrix &R, matrix &Out)
{
matrix Y;
vector v1, v2;
v1.Init(R.Cols());

Y = ArgMax(Out);
v2.Init(Y.Cols());

ulong count1, count2;
double N_samples, N_hits, Accur;
N_samples = R.Rows();
N_hits = 0;

for(int i=0;i<R.Rows();i++)
  {for(int j=0;j<R.Cols();j++)
     {v1[j] = R[i][j];
      v2[j] = Y[i][j];}
   count1 = v1.ArgMax();
   count2 = v2.ArgMax();
   if(count1 == count2) N_hits = N_hits + 1.0;
   
  }
  
Accur = N_hits/N_samples;
return Accur;
}

double Metrics::MAPE(matrix &R, matrix &Y)
{
matrix Err;
Err = R - Y; 
Err = Err / R; 
Err = MathAbs(Err);

double error; 
error = Err.Mean();

return error; 
}