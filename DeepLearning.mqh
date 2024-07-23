#include <DeepLearningLibrary.mqh>

class DeepLearning
  {
public:
   //-------------------------------
   // Define the cost functions and its derivatives
   class Loss;
   // Define metrics for evaluation
   class Metrics;
   
   //-------------------------------
   // Dense Connected Layer
   class DenseLayer;
   // Activation Function Layer 
   class ActivationLayer;
   // Softmax function Layer
   class SoftmaxLayer;
   // Dropout Layer to Enhance Overffiting
   class DropoutLayer;

   //--------------------------------
   // Convolutional Neural Net Layer
   class ConvolutionalLayer;
   // Max Pooling Layer
   class MaxPoolingLayer;
   // Flatten Layer
   class FlattenLayer;
   // Sum Convolutional Layer
   class SumConvLayer;

   //--------------------------------
   //Long Short-Term Memory Layer
   class LSTMLayer;
   // Bidirectional LSTM layer
   class BiLSTMLayer;
   
   
   //Methods
   virtual matrix    Output(matrix &X)                {return X*0;   }
   virtual matrix    GradDescent(matrix &Ey)          {return Ey*0;  }
   virtual void      Update(void)                     {              }
   virtual void      SaveWeights(int k,string IAname) {              }
   virtual void      LoadWeights(int k,string IAname) {              }
   virtual void      SetDrop(double Drop)             {              }
   virtual void      SetAdam(double B1, double B2, double Alph) {    }
   
   
   
   //=============================
   matrix   InitWeights(matrix &M);
   matrix   ZeroMatrix(matrix &M);
   void     SaveMatrix(matrix &M, string M_name);
   matrix   LoadMatrix(string M_name);
   matrix   Concatenate(matrix &X, matrix &H);
   
   //Convolution 
   matrix VertConvV(matrix &A, matrix &B);
   matrix VertConvF(matrix &A, matrix &B);
   matrix VertInv(matrix &A);
   
   matrix HorConvV(matrix &A, matrix &B);
   matrix HorConvF(matrix &A, matrix &B);
   matrix HorInv(matrix &A);
   
   //Activation
   matrix Sig(matrix &X);
   matrix Tanh(matrix &X);
   matrix ReLU(matrix &X);
   
   matrix dSig(matrix &X);
   matrix dTanh(matrix &X);
   matrix dReLU(matrix &X);
   
   //ADAM optimizer
   matrix AdamM(matrix &m, matrix &dX,double beta1);
   matrix AdamV(matrix &v, matrix &dX,double beta2);
   matrix Adam(double it, matrix &m, matrix &v,double beta1, double beta2, double alpha);
   
   
  };
 
//+------------------------------------------------------------------+
//|   Deep Learning Methodes                                         |
//+------------------------------------------------------------------+
matrix DeepLearning::InitWeights(matrix &M)
{
   matrix W;
   W = M;
   for(int i=0;i<W.Rows();i++)
     {for(int j=0;j<W.Cols();j++)
        {W[i][j] = (2.0*(MathRand()/32766.0) -1.0);}}
         
return W;
}
matrix DeepLearning::ZeroMatrix(matrix &M)
{
for(int i=0;i<M.Rows();i++)
  {for(int j=0;j<M.Cols();j++)
     {M[i][j] = 0;}}
      
return M;
}
void DeepLearning::SaveMatrix(matrix &M,string M_name)
{
   //transforma a matrix M num vetor de strings
   ulong Srows , SCols;
   Srows = M.Rows();
   SCols = M.Cols();
   string csv_name;
   csv_name = M_name;
   
   string V[];
   ArrayResize(V,Srows);
   
   //Zera o vetor de strings
   for(int i=0;i<ArraySize(V);i++)
     {V[i] = NULL;}
      
   //Prepara o vetor com as classes 

   for(int i=0;i<Srows;i++)
     {for(int j=0;j<SCols;j++)
         {
         if(j == SCols-1) V[i] = V[i] + DoubleToString(M[i][j]);
         else V[i] = V[i] + DoubleToString(M[i][j]) + ",";}}     
   
   //Abre o arquivo para ser escrito
   int h=FileOpen(csv_name,FILE_WRITE|FILE_ANSI|FILE_CSV);
   //Se o arquivo não é aberto devidamente o handle é inválido
   if(h==INVALID_HANDLE) Alert("Error opening file");
   
   for(int i=0;i<Srows;i++)
      {
      FileWrite(h,V[i]);
      }
   FileClose(h);
}
matrix DeepLearning::LoadMatrix(string M_name)
{
   //Le apenas a primeira linha para saber o número de colunas
   string L1;
   string csv_name;
   csv_name = M_name;
   //Abre o arquivo para ser lido
   int h1=FileOpen(csv_name,FILE_READ|FILE_ANSI|FILE_TXT);
   //Se o arquivo não é aberto devidamente o handle é inválido
   if(h1==INVALID_HANDLE)   Alert("Error opening file");
   L1 = FileReadString(h1);
   FileClose(h1);
   
   //L1 possui agora a primeira linha da matriz
   //Lê quantas colunas são pelo número de vírgulas
   
   int num_columns = 1; 
   
   for(int i=0;i<L1.Length();i++)
     {
      if(L1.Substr(i,1) == ",") num_columns++;
     }
   
   //Abre o arquivo para ser lido
   int h=FileOpen(csv_name,FILE_READ|FILE_ANSI|FILE_CSV,",");
   //Se o arquivo não é aberto devidamente o handle é inválido
   if(h==INVALID_HANDLE)   Alert("Error opening file");

   string read_x;
   string m[]; //Vetor que receberá os dados
   int    m_size = 0;
   
   matrix A;   // Matriz que retornará com os dados
   int A_size = 0;
   //Começa com a leitura da primeira linha
   
   while(!FileIsEnding(h))
   {
      ArrayResize(m,m_size+1);
      read_x = FileReadString(h);   // Lê o conteudo até a virgula é passa pra próxima
      m[m_size] = read_x;
   if(!FileIsEnding(h)) m_size++;  
   }
   FileClose(h);
   
   int num_rows;
   num_rows = (m_size + 1)/num_columns;
   
   if(((m_size +1)% num_columns) != 0 )   Alert("Error the matrix data is incomplete");
   else
   {
   
   //Preparar a Matriz A
   A.Init(num_rows,num_columns);
   
   for(int i=0;i<num_rows;i++)
      {for(int j=0;j<num_columns;j++)
        {A[i][j] = StringToDouble(m[i * num_columns + j]);}}         
   //==========
   }
return A;
}
matrix DeepLearning::Concatenate(matrix &X,matrix &H)
{
if(X.Cols() != H.Cols()) Alert("The number of Cols of X and H must be equal");

matrix M;
M.Init(X.Rows() + H.Rows(),X.Cols());

ulong lim;
lim = X.Rows();

for(int i=0;i<M.Rows();i++)
  {for(int j=0;j<M.Cols();j++)
     {if(i < lim) M[i][j] = X[i][j];
      if(i >= lim) M[i][j] = H[i-lim][j];}}
      
return M;
}

//+------------------------------------------------------------------+
//|    Convolutional Methodes                                        |
//+------------------------------------------------------------------+

matrix DeepLearning::VertConvV(matrix &A,matrix &B)
{
if(A.Cols() != B.Cols())
  {Alert("matrices with different number of Columns");
  return A*0;
  }
matrix U,D,C;
U = A;
D = B;
if(U.Rows() < D.Rows())
  {C = U;
   U = D;
   D = C;
   }
D = VertInv(D);

matrix Conv; 
Conv.Init(U.Rows()-D.Rows()+1,U.Cols());

//Zera a matriz
for(int i=0;i<Conv.Rows();i++)
  {for(int j=0;j<Conv.Cols();j++)
     {Conv[i][j] = 0;}}
      

for(int i=0;i<Conv.Rows();i++)
  {for(int j=0;j<Conv.Cols();j++)
     {for(int k=0;k<D.Rows();k++)
        {Conv[i][j] = Conv[i][j] + U[i+k][j]*D[k][j];}}}
      
return Conv;
}
matrix DeepLearning::HorConvV(matrix &A,matrix &B)
{
if(A.Rows() != B.Rows())
  {Alert("matrices with different number of Rows");
  return A*0;
  }
matrix U,D,C;
U = A;
D = B;
if(U.Cols() < D.Cols())
  {C = U;
   U = D;
   D = C;
   }
D = HorInv(D);

matrix Conv; 
Conv.Init(U.Rows(), U.Cols() - D.Cols() + 1);

//Zera a matriz
for(int i=0;i<Conv.Rows();i++)
  {for(int j=0;j<Conv.Cols();j++)
     {Conv[i][j] = 0;}}
      

for(int i=0;i<Conv.Rows();i++)
  {for(int j=0;j<Conv.Cols();j++)
     {for(int k=0;k<D.Cols();k++)
        {Conv[i][j] += U[i][j+k]*D[i][k];}}}
      
return Conv;
}
matrix DeepLearning::VertConvF(matrix &A,matrix &B)
{
if(A.Cols() != B.Cols())
  {Alert("matrices with different number of Columns");
  return A*0;
  }

matrix U,D,E;

U = A;
D = B;

if(U.Rows() < D.Rows())
  {E = U;
   U = D;
   D = E;
   }
matrix Conv;
Conv.Init(U.Rows()+2*D.Rows()-2,U.Cols());

//Zera a matriz
for(int i=0;i<Conv.Rows();i++)
  {for(int j=0;j<Conv.Cols();j++)
     {Conv[i][j] = 0;}}

ulong c = B.Rows()-1;
for(int i=c;i<Conv.Rows()-c;i++)
  {for(int j=0;j<Conv.Cols();j++)
     {
      Conv[i][j] = U[i-c][j];
     }
   
  }

matrix CF;
   CF= VertConvV(Conv,D);
return CF;
}
matrix DeepLearning::HorConvF(matrix &A,matrix &B)
{
if(A.Rows() != B.Rows())
  {Alert("matrices with different number of Rows");
  return A*0;
  }

matrix U,D,E;

U = A;
D = B;

if(U.Rows() < D.Rows())
  {E = U;
   U = D;
   D = E;
   }
matrix Conv;
Conv.Init(U.Rows(),U.Cols() + 2*D.Cols() -2);

//Zera a matriz
for(int i=0;i<Conv.Rows();i++)
  {for(int j=0;j<Conv.Cols();j++)
     {Conv[i][j] = 0;}}

ulong c = B.Cols()-1;

for(int i=0;i<Conv.Rows();i++)
  {for(int j=c;j<Conv.Cols()-c;j++)
     {
      Conv[i][j] = U[i][j-c];
     }
   
  }

matrix CF;
   CF= HorConvV(Conv,D);
return CF;
}
matrix DeepLearning::VertInv(matrix &A)
{
matrix B;
B.Init(A.Rows(),A.Cols());
for(int i=0;i<A.Rows();i++)
  {for(int j=0;j<A.Cols();j++)
     {B[i][j] = A[A.Rows()-i-1][j];}}


return B;
}
matrix DeepLearning::HorInv(matrix &A)
{
matrix B;
B.Init(A.Rows(),A.Cols());
for(int i=0;i<A.Rows();i++)
  {for(int j=0;j<A.Cols();j++)
     {B[i][j] = A[i][A.Cols()-j-1];}}


return B;
}

//+------------------------------------------------------------------+
//|    Activation Methodes                                           |
//+------------------------------------------------------------------+

matrix DeepLearning::Sig(matrix &X)
{
matrix M;
M = X;
for(int i=0;i<M.Rows();i++)
  {for(int j=0;j<M.Cols();j++)
     {M[i][j] = 1.0/(1.0 + MathExp((-1)*M[i][j]));}}
     
return M;
      
}
matrix DeepLearning::Tanh(matrix &X)
{
matrix M;
M = X;
for(int i=0;i<M.Rows();i++)
  {for(int j=0;j<M.Cols();j++)
     {M[i][j] = (MathExp(M[i][j])-MathExp((-1.0)*M[i][j]))/(MathExp(M[i][j])+MathExp((-1.0)*M[i][j]));}}
     
return M;
}
matrix DeepLearning::ReLU(matrix &X)
{
matrix M;
M = X; 
for(int i=0;i<M.Rows();i++)
  {for(int j=0;j<M.Cols();j++)
     {if(M[i][j] > 0) M[i][j] = M[i][j];
      if(M[i][j] <=0) M[i][j] = 0.01*M[i][j];}}

return M;     
}

matrix DeepLearning::dSig(matrix &X)
{
matrix M;
M = X; 

M = Sig(M);
for(int i=0;i<M.Rows();i++)
  {for(int j=0;j<M.Cols();j++)
     {M[i][j] = M[i][j]*(1.0 - M[i][j]);}}

return M;  
}
matrix DeepLearning::dTanh(matrix &X)
{
matrix M;
M = X; 

M = Tanh(M);
for(int i=0;i<M.Rows();i++)
  {for(int j=0;j<M.Cols();j++)
     {M[i][j] = (1.0 - M[i][j]*M[i][j]);}}

return M;  
}
matrix DeepLearning::dReLU(matrix &X)
{
matrix M;
M = X;
for(int i=0;i<M.Rows();i++)
  {for(int j=0;j<M.Cols();j++)
     {if(M[i][j] > 0) M[i][j] = 1;
      if(M[i][j] <= 0) M[i][j] = 0.01;}}
      
return M;
}
//+------------------------------------------------------------------+
//|   Optimizers                                                     |
//+------------------------------------------------------------------+
matrix DeepLearning::AdamM(matrix &m, matrix &dX,double beta1)
{
matrix mt;
mt.Init(dX.Rows(),dX.Cols());
mt = m * beta1;
mt = mt + dX * (1-beta1);
return mt;
}
matrix  DeepLearning::AdamV(matrix &v, matrix &dX,double beta2)
{
matrix vt;
vt = beta2*v;
vt = vt + dX * dX * (1-beta2);
return vt; 
}
matrix DeepLearning::Adam(double it, matrix &m, matrix &v,double beta1,double beta2, double alpha)
{
matrix D, mt, vt; 

mt = m * (1/(1-MathPow(beta1,it)));
vt = v * (1/(1-MathPow(beta2,it)));

vt = MathSqrt(vt) + 1e-8; 
D = m / vt;
D = D * alpha;
return D; 
}

#include <Layers\LossLayer.mqh>
#include <Layers\MetricsLayer.mqh>

#include <Layers\DenseLayer.mqh>
#include <Layers\ActivationLayer.mqh>
#include <Layers\DropoutLayer.mqh>

#include <Layers\ConvolutionalLayer.mqh>
#include <Layers\MaxPoolingLayer.mqh>
#include <Layers\FlattenLayer.mqh>
#include <Layers\SumConvLayer.mqh>

#include <Layers\LSTMLayer.mqh>
#include <Layers\BiLSTMLayer.mqh>









