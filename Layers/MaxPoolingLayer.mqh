//+------------------------------------------------------------------+
//|   Max Pooling                                                    |
//+------------------------------------------------------------------+

class MaxPoolingLayer : public DeepLearning
  {
public:
   //Initializes parameters
   void   InitLayer( int stride, CONV_DIR direction);
   //Calculates the output of the litter from an input
   //The output dimension is  Ceil(N_entries / stride)
   virtual matrix Output(matrix &X);
   //Propagates the Error
   virtual matrix GradDescent(matrix &Ey);

private:
   //Stride
   int S;
   //Direction
   CONV_DIR DIR; 
   //Matrix with the index of the maximum values
   matrix Max;
   //Number of steps 
   int Steps; 
   
   //
  };

//+------------------------------------------------------------------+
//|    Max Pooling                                                              |
//+------------------------------------------------------------------+
void   MaxPoolingLayer::InitLayer( int stride, CONV_DIR direction)
{
S = stride;
DIR = direction;

}
matrix MaxPoolingLayer::Output(matrix &X)
{
matrix M,MP,Mx;

if(DIR == VERT)
  {
   M.Init(S,X.Cols());
   
   int N_slides,N_steps,count;
   Steps = X.Rows();
   
   double N_out; 
   N_out = X.Rows();
   N_out = N_out/S;
   N_out = MathCeil(N_out);
   
   N_slides = N_out;
   MP.Init(N_slides,X.Cols());
   Mx.Init(N_slides,X.Cols());
   
   //A ideia é selecionar a matriz correspondente ao stride
   //E tomar o elemento máximo de cada coluna dessa matriz
   vector v; 
   for(int k=0;k<N_slides;k++)
      {for(int i=0;i<M.Rows();i++)
        {for(int j=0;j<M.Cols();j++)
           {if(k*S+i < Steps)  M[i][j] = X[k*S+i][j];
            if(k*S+i >= Steps) M[i][j] = -1e12;}}
   
      v = M.ArgMax(0);
      for(int i=0;i<MP.Cols();i++)
        {count = v[i];
         MP[k][i] = M[count][i];
         Mx[k][i] = v[i];
        }
      }
         
   Max = Mx;
   }
if(DIR == HORZ)
  {
   M.Init(X.Rows(),S);
   
   int N_slides,N_steps,count;
   Steps = X.Cols();
   
   double N_out; 
   N_out = X.Cols();
   N_out = N_out/S;
   N_out = MathCeil(N_out);
   
   N_slides = N_out;
   
   MP.Init(X.Rows(),N_slides);
   Mx.Init(X.Rows(),N_slides);
   
   //A ideia é selecionar a matriz correspondente ao stride
   //E tomar o elemento máximo de cada coluna dessa matriz
   vector v; 
   for(int k=0;k<N_slides;k++)
      {for(int i=0;i<M.Rows();i++)
        {for(int j=0;j<M.Cols();j++)
           {if(k*S+j < Steps)  M[i][j] = X[i][k*S+j];
            if(k*S+j >= Steps) M[i][j] = -1e12;}}
   
      v = M.ArgMax(1);
      for(int i=0;i<MP.Rows();i++)
        {count = v[i];
         MP[i][k] = M[i][count];
         Mx[i][k] = v[i];
        }
      }
         
   Max = Mx;
   }
return MP;
}

matrix MaxPoolingLayer::GradDescent(matrix &Ey)
{

   matrix Ex;
if(DIR == VERT)
{
   Ex.Init(Steps,Ey.Cols());
   
   for(int k=0;k<Steps;k++)
      {for(int j=0;j<Ey.Cols();j++)
        {for(int i=0;i<S;i++)
           {if(k*S+i < Steps)
              {if(i == Max[k][j])
              { Ex[k*S+i][j] = Ey[k][j];}
               if(i != Max[k][j]) 
               {Ex[k*S+i][j] = 0;}
               
       }}}}
}
if(DIR == HORZ)
{
   Ex.Init(Ey.Rows(),Steps);
   
   for(int k=0;k<Steps;k++)
      {for(int i=0;i<Ey.Rows();i++)
        {for(int j=0;j<S;j++)
           {if(k*S+j < Steps)
              {if(j == Max[i][k])
              { Ex[i][k*S+j] = Ey[i][k];}
               if(j != Max[i][k]) 
               {Ex[i][k*S+j] = 0;}
               
       }}}}
}        
return Ex;
}
