//+------------------------------------------------------------------+
//|   Flatten                                                        |
//+------------------------------------------------------------------+

class FlattenLayer : public DeepLearning
  {
public:
   //Calculates the output of the litter from an input
   //Transforms each row of matrix X into a column and then puts a column
   //below column EX: {{1,2},{3,4}} => {{1},{2},{3},{4}}
   virtual matrix Output(matrix &X);
   //Propagates the Error
   virtual matrix GradDescent(matrix &Ey);
   //Updates the weights 
private: 
   int N_steps;
   int N_entries;
  };

//+------------------------------------------------------------------+
//|    Flatten Layer                                                 |
//+------------------------------------------------------------------+

matrix FlattenLayer::Output(matrix &X)
{
N_entries = X.Rows();
N_steps = X.Cols();
matrix Y;
Y.Init(N_entries*N_steps,1);
for(int i=0;i<N_entries;i++)
  {for(int j=0;j<N_steps;j++)
     {Y[i*N_entries+j][0] = X[i][j];}}

return Y;
}  
matrix FlattenLayer::GradDescent(matrix &Ey)
{
matrix Ex;
Ex.Init(N_entries,N_steps);

for(int i=0;i<N_entries;i++)
  {for(int j=0;j<N_steps;j++)
     {Ex[i][j]= Ey[i*N_entries+j][0];}}

return Ex;
}

