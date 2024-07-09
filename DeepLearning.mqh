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
   class softmaxLayer;
   // Dropout Layer to Enhance Overffiting
   class DropoutLayer;

   //--------------------------------
   // Convolutional Neural Net Layer
   class ConvolutionalLayer;
   // Max Pooling Layer
   class MaxPoolingLayer;
   // Flatten Layer
   class FlattenLayer;

   //--------------------------------
   //Long Short-Term Memory Layer
   class LSTMLayer;
   // Bidirectional LSTM layer
   class biLSTMLayer;
   
   
   //Methods
   virtual matrix    Output(matrix &X)               {return X*0;   }
   virtual matrix    GradDescent(matrix &Ey)          {return Ey*0;  }
   virtual void      Update(void)                     {              }
   virtual void      SaveWeights(int k,string IAname) {              }
   virtual void      LoadWeights(int k,string IAname) {              }
   virtual void      SetDrop(double Drop)             {              }
   virtual void      SetAdam(double B1, double B2, double Alph) {   }
   
   
   
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
//|   Sum Convolutional                                              |
//+------------------------------------------------------------------+

class SumConvLayer : public DeepLearning
  {
public :
   //Inicializa os kernels e o bias da camada
   //1<= N_outputs <= N_entries; N_entries-N_outputs+1 =KernelSize.  
   void InitLayer(int N_steps, int N_entries, int N_outputs, double LR, int N_Conv, CONV_DIR direction = HORZ, ActFunction af = TANH, Optim Op = STD);
   //Calcula a saída da camada a partir de uma entrada
   virtual matrix Output(matrix &X);
   //Propaga o Erro 
   virtual matrix GradDescent(matrix &Ey);
   //Atualiza os pesos
   virtual void   Update(void);
   //Salvar os pesos, k é o indice da camada;
   virtual void   SaveWeights(int k,string IAname);
   //Carregar os pesos
   virtual void   LoadWeights(int k,string IAname);

private:
   //Number of convolutions
   int N_C;
   //Kernel da Camada
   matrix K[];
   //Gradiente do Kernel
   matrix dK[];
   //Bias da camada
   matrix B[];
   //Gradiente do Bias
   matrix dB[];
   //Entrada da Camada de ativação
   matrix Xact[];
   //Peso auxiliar
   matrix C[];
   //gradiente do peso auxiliar 
   matrix dC[];
   //Entrada da Camada convolucional
   matrix Xc;
   //Taxa de aprendizagem
   double N;
   //Método de otimização
   Optim OP;
   //Direção da convolução
   CONV_DIR DIR;
   //Função de Ativação
   ActFunction AF;
   
   //ADAM 
   //===============
   //Contador de iterações
   ulong it; 
   //m
   matrix mK[], mB[], mC[];
   //v
   matrix vK[], vB[], vC[];
   //hiper parâmetros
   double beta1, beta2, alpha; 
  };
//+------------------------------------------------------------------+
//|   LSTM                                                           |
//+------------------------------------------------------------------+
class LSTMLayer : public DeepLearning
  {
public:
   //Initializes layer weights and bias
   void InitLayer(int N_steps, int N_entries,int N_hidden, int N_outputs, double LR, Optim Op = STD);
   //Calculates the output of the layer from an input
   virtual matrix Output(matrix &X);
   //Propagates the Error
   virtual matrix GradDescent(matrix &Ey);
   //Update the weights
   virtual void   Update(void);
   //Save the weights, k is the layer index;
   virtual void   SaveWeights(int k,string IAname);
   //Load the weights 
   virtual void   LoadWeights(int k,string IAname);
   
   //Changing ADAM parameters
   virtual void   SetAdam(double B1, double B2, double Alph); 
   
   
private:
   //Learning Rate
   double N;
   //Optimizatiom
   Optim OP;
   //Number of time steps
   ulong N_ts;
   //Dimension of Hidden State
   ulong N_H;
   //Number of entries
   ulong N_ins;
   //Number of outputs
   ulong N_outs;
   
   //Forget gate
   matrix Wf,Bf,Zf[],
          dWf,dBf;
   //Input gate
   matrix Wi,Bi,Zi[],
          dWi,dBi;
   //Output gate
   matrix Wo,Bo,Zo[],
          dWo,dBo;
   //Candidate gate
   matrix Wg,Bg,Zg[],
          dWg,dBg;
   //output
   matrix Wy,By,Zy[],
          dWy,dBy;
   //Hidden state
   matrix H[];
   //Long term memory
   matrix C[];
   //Input
   matrix x[],x_h[];
  
   //ADAM 
   //===============
   //Iteration counter
   ulong it; 
   //m
   matrix mWy, mWf, mWo, mWi, mWg;
   matrix mBy, mBf, mBo, mBi, mBg;
   //v
   matrix vWy, vWf, vWo, vWi, vWg;
   matrix vBy, vBf, vBo, vBi, vBg;
   
   //hiperparameters 
   double beta1, beta2, alpha; 
  
  
  };
//+------------------------------------------------------------------+
//|   biLSTM Layer                                                   |
//+------------------------------------------------------------------+

class biLSTMLayer : public DeepLearning
  {
public:
   //Initializes layer weights and bias
   void InitLayer(int N_steps, int N_entries,int N_hidden, int N_outputs, double LR, Optim Op = STD);
   //Calculates the output of the layer from an input
   virtual matrix Output(matrix &X);
   //Propagates the error
   virtual matrix GradDescent(matrix &Ey);
   //Updates the error
   virtual void   Update(void);
   //Save weights, k is the layer index;
   virtual void   SaveWeights(int k,string IAname);
   //load the weights
   virtual void   LoadWeights(int k,string IAname);
   
   
   //ADAM
   virtual void   SetAdam(double B1, double B2, double Alph);
   
   
private:
   //Learning Rate
   double N;
   //Optimizatiom
   Optim OP;
   //Number of time steps
   ulong N_ts;
   //Dimension of Hidden State
   ulong N_H;
   //Number of entries
   ulong N_ins;
   //Number of outputs
   ulong N_outs;
   
   //From Past to Future:
   //Forget gate
   matrix Wfp,Bfp,Zfp[],
          dWfp,dBfp;
   //Input gate
   matrix Wip,Bip,Zip[],
          dWip,dBip;
   //Output gate
   matrix Wop,Bop,Zop[],
          dWop,dBop;
   //Candidate gate
   matrix Wgp,Bgp,Zgp[],
          dWgp,dBgp;
   //saída
   matrix Wyp,Byp,Zyp[],
          dWyp,dByp;
   //Estado Oculto
   matrix Hp[];
   //Memória de Longo Prazo
   matrix Cp[];
   //Entrada
   matrix xp[],x_hp[];
  
  
   //From Future to Past
   //Forget gate
   matrix Wff,Bff,Zff[],
          dWff,dBff;
   //Input gate
   matrix Wif,Bif,Zif[],
          dWif,dBif;
   //Output gate
   matrix Wof,Bof,Zof[],
          dWof,dBof;
   //Candidate gate
   matrix Wgf,Bgf,Zgf[],
          dWgf,dBgf;
   //saída
   matrix Wyf,Byf,Zyf[],
          dWyf,dByf;
   //Estado Oculto
   matrix Hf[];
   //Memória de Longo Prazo
   matrix Cf[];
   //Entrada
   matrix xf[],x_hf[];
   
   //ADAM 
   //===============
   //Contador de iterações
   ulong it; 
   //m
   matrix mWyp, mWfp, mWop, mWip, mWgp;
   matrix mByp, mBfp, mBop, mBip, mBgp;
   //v
   matrix vWyp, vWfp, vWop, vWip, vWgp;
   matrix vByp, vBfp, vBop, vBip, vBgp;
   
   matrix mWyf, mWff, mWof, mWif, mWgf;
   matrix mByf, mBff, mBof, mBif, mBgf;
   //v
   matrix vWyf, vWff, vWof, vWif, vWgf;
   matrix vByf, vBff, vBof, vBif, vBgf;
   
   //hiper parâmetros
   double beta1, beta2, alpha; 
  
  
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
//+------------------------------------------------------------------+
//|   Sum Convolutional                                              |
//+------------------------------------------------------------------+ 
void SumConvLayer::InitLayer(int N_steps, int N_entries, int N_outputs, double LR,int N_Conv, CONV_DIR direction = HORZ, ActFunction af = TANH, Optim Op = STD)
{
   N_C = N_Conv;
   AF = af;
   
   double factor; 
   factor = N_Conv;
   factor = 1 / factor; 
   
   ArrayResize(K,N_C);
   ArrayResize(dK,N_C);
   
   ArrayResize(B,N_C);
   ArrayResize(dB,N_C);
   
   ArrayResize(Xact,N_C);
   ArrayResize(C,N_C);
   ArrayResize(dC,N_C);
   
for(int i=0;i<N_C;i++)
  {
      if(direction == VERT)
      {  K[i].Init(N_steps-N_outputs+1,N_entries);
         B[i].Init(N_steps-K[i].Rows()+1,N_entries);
         C[i].Init(B[i].Rows(),B[i].Cols());
         K[i] = InitWeights(K[i]);
         B[i] = InitWeights(B[i]);
         C[i] = InitWeights(C[i]) * factor;}
   
      if(direction == HORZ)
      {  K[i].Init(N_entries,N_steps-N_outputs + 1);
         B[i].Init(N_entries,N_steps - K[i].Rows() +1);
         C[i].Init(B[i].Rows(),B[i].Cols());
         K[i] = InitWeights(K[i]);
         B[i] = InitWeights(B[i]);
         C[i] = InitWeights(C[i]) * factor;}
   }
   N = LR;
   OP = Op;

   if(OP == ADAM)
     {
      it = 0;
      beta1 = 0.9;
      beta2 = 0.999;
      alpha = N;
      
      ArrayResize(mK,N_C);
      ArrayResize(vK,N_C);
      
      ArrayResize(mB,N_C);
      ArrayResize(vB,N_C);
      
      ArrayResize(mC,N_C);
      ArrayResize(vC,N_C);
      
      for(int i=0;i<N_C;i++)
      {
         mK[i].Init(K[i].Rows(),K[i].Cols());
         vK[i].Init(K[i].Rows(),K[i].Cols());  
         mK[i] = ZeroMatrix(mK[i]);
         vK[i] = ZeroMatrix(vK[i]);
         
         mB[i].Init(B[i].Rows(),B[i].Cols());
         vB[i].Init(B[i].Rows(),B[i].Cols());
         mB[i] = ZeroMatrix(mB[i]);
         vB[i] = ZeroMatrix(vB[i]);
         
         mC[i].Init(C[i].Rows(),C[i].Cols());
         vC[i].Init(C[i].Rows(),C[i].Cols());
         mC[i] = ZeroMatrix(mC[i]);
         vC[i] = ZeroMatrix(vC[i]);
      }
     }
}
matrix SumConvLayer::Output(matrix &X)
{
matrix Out, Y[];
Xc = X;
ArrayResize(Y,N_C);
for(int i=0;i<N_C;i++)
  {
   if(DIR == VERT) Y[i] = VertConvV(X,K[i]) + B[i];
   if(DIR == HORZ) Y[i] = HorConvV(X,K[i]) + B[i];
   }
Out.Init(B[0].Rows(),B[0].Cols());

//Gravar a entrada
for(int i=0;i<N_C;i++)
  {Xact[i] = Y[i];}

//Zerar a saída 
for(int i=0;i<Out.Rows();i++)
  {for(int j=0;j<Out.Cols();j++)
     {Out[i][j] = 0;}}

//Aplicar a função de ativação; 
for(int i=0;i<N_C;i++)
  {if(AF == SIGMOID) Y[i] = Sig(Y[i]);
   if(AF == TANH)    Y[i] = Tanh(Y[i]);
   if(AF == RELU)    Y[i] = ReLU(Y[i]);}

//Soma tudo
for(int i=0;i<N_C;i++)
  {Out +=Y[i] * C[i];}
  
return Out;
}

matrix SumConvLayer::GradDescent(matrix &Ey)
{
matrix Ex,dphi[]; 
ArrayResize(dphi,N_C);
Ex.Init(Xc.Rows(),Xc.Cols());

//Peso auxiliar
for(int i=0;i<N_C;i++)
   {if(AF == SIGMOID) dC[i] = Sig(Xact[i]) * Ey;
   if(AF == TANH) dC[i] = Tanh(Xact[i]) * Ey;
   if(AF == RELU) dC[i] = ReLU(Xact[i]) * Ey;}

//Calcular a derivada da função de ativação
for(int i=0;i<N_C;i++)
  {if(AF == SIGMOID) dphi[i] = dSig(Xact[i]) * Ey *C[i];
   if(AF == TANH) dphi[i] = dTanh(Xact[i]) * Ey *C[i];
   if(AF == RELU) dphi[i] = dReLU(Xact[i]) * Ey *C[i];}

//Zerar o gradiente da entrada
for(int i=0;i<Ex.Rows();i++)
  {for(int j=0;j<Ex.Cols();j++)
     {Ex[i][j] = 0;}}

matrix Inv;
//Calcular as variações dos pesos
for(int i=0;i<N_C;i++)
  {if(DIR == VERT)
     {Inv = VertInv(K[i]);
      Ex += VertConvF(Inv,dphi[i]);
      Inv = VertInv(Xc);
      dK[i] = VertConvV(Inv,dphi[i]);
      dB[i] = dphi[i];}
      
   
   if(DIR == HORZ)
     {Inv = HorInv(K[i]);
      Ex += HorConvF(Inv,dphi[i]);
      Inv = HorInv(Xc);
      dK[i] = HorConvV(Inv,dphi[i]);
      dB[i] = dphi[i];} 
  } 

return Ex;   
}


void   SumConvLayer::Update(void)
{
if(OP == STD)
   {for(int i=0;i<N_C;i++)
     {K[i] = K[i] - dK[i]*N;
      B[i] = B[i] - dB[i]*N;
      C[i] = C[i] - dC[i]*N;
     }
   }
if(OP == ADAM)
  {
   it +=1;
   for(int i=0;i<N_C;i++)
     {
      mK[i] = AdamM(mK[i],dK[i],beta1);
      vK[i] = AdamV(vK[i],dK[i],beta2);
      K[i] = K[i] - Adam(it,mK[i],vK[i],beta1,beta2,alpha);
      
      mB[i] = AdamM(mB[i],dB[i],beta1);
      vB[i] = AdamV(vB[i],dB[i],beta2);
      B[i] = B[i] - Adam(it,mB[i],vB[i],beta1,beta2,alpha);
      
      mC[i] = AdamM(mC[i],dC[i],beta1);
      vC[i] = AdamV(vC[i],dC[i],beta2);
      C[i] = C[i] - Adam(it,mC[i],vC[i],beta1,beta2,alpha);
     }
  }
}

void   SumConvLayer::SaveWeights(int k,string IAname)
{
for(int i=0;i<N_C;i++)
  {
   string csv_name;
   csv_name = IAname + "\KLayer" + IntegerToString(k)+ IntegerToString(i);
   SaveMatrix(K[i],csv_name);
   csv_name = IAname + "\BLayer" + IntegerToString(k)+ IntegerToString(i);
   SaveMatrix(B[i],csv_name);
   csv_name = IAname + "\CLayer" + IntegerToString(k)+ IntegerToString(i);
   SaveMatrix(C[i],csv_name);
  }
}

void   SumConvLayer::LoadWeights(int k,string IAname)
{
for(int i=0;i<N_C;i++)
   {  
   string csv_name;
   csv_name = IAname + "\KLayer" + IntegerToString(k) + IntegerToString(i);
   K[i] = LoadMatrix(csv_name);
   csv_name = IAname + "\BLayer" + IntegerToString(k) + IntegerToString(i);
   B[i] = LoadMatrix(csv_name);
   csv_name = IAname + "\BLayer" + IntegerToString(k) + IntegerToString(i);
   C[i] = LoadMatrix(csv_name);
   }
}
//+------------------------------------------------------------------+
//|   LSTM                                                           |
//+------------------------------------------------------------------+

void LSTMLayer::InitLayer(int N_steps, int N_entries,int N_hidden, int N_outputs, double LR, Optim Op = STD)
{
//Hiperparameters 
N = LR;
OP = Op;
N_ts = N_steps;
N_ins = N_entries;
N_outs = N_outputs;
N_H = N_hidden;

//parâmetro de Xavier para inicialização. 
double xavier_;
xavier_ = N_ins + N_outs;
xavier_ = MathSqrt(6/xavier_);

//forget gate
Wf.Init(N_H,N_H + N_ins);
dWf.Init(N_H,N_H + N_ins);
Wf = InitWeights(Wf)*xavier_;

Bf.Init(N_H,1);
dBf.Init(N_H,1);
Bf = InitWeights(Bf)*xavier_;

ArrayResize(Zf,N_ts);
     
//input gate
Wi.Init(N_H,N_H + N_ins);
dWi.Init(N_H,N_H + N_ins);
Wi = InitWeights(Wi)*xavier_;

Bi.Init(N_H,1);
dBi.Init(N_H,1);
Bi = InitWeights(Bi)*xavier_;

ArrayResize(Zi,N_ts);
//output gate
Wo.Init(N_H,N_H + N_ins);
dWo.Init(N_H,N_H + N_ins);
Wo = InitWeights(Wo)*xavier_;

Bo.Init(N_H,1);
dBo.Init(N_H,1);
Bo = InitWeights(Bo)*xavier_;

ArrayResize(Zo,N_ts);     
// candidate gate
Wg.Init(N_H,N_H + N_ins);
dWg.Init(N_H,N_H + N_ins);
Wg = InitWeights(Wg)*xavier_;

Bg.Init(N_H,1);
dBg.Init(N_H,1);
Bg = InitWeights(Bg)*xavier_;

ArrayResize(Zg,N_ts);
//Output
Wy.Init(N_outs,N_H);
dWy.Init(N_outs,N_H);
Wy = InitWeights(Wy)*xavier_;


By.Init(N_outs,1);
dBy.Init(N_outs,1);
By = InitWeights(By)*xavier_;

ArrayResize(Zy,N_ts);

//Estado Oculto
ArrayResize(H,N_ts+1);

//Memória de longo prazo
ArrayResize(C,N_ts+1);

//Entrada 
ArrayResize(x,N_ts);
ArrayResize(x_h,N_ts);

   if(OP == ADAM)
     {
      it = 0;
      beta1 = 0.9;
      beta2 = 0.999;
      alpha = N;
      //Saída
      mWy.Init(Wy.Rows(),Wy.Cols());
      vWy.Init(Wy.Rows(),Wy.Cols());  
      mWy = ZeroMatrix(mWy);
      vWy = ZeroMatrix(vWy);
      
      mBy.Init(By.Rows(),By.Cols());
      vBy.Init(By.Rows(),By.Cols());
      mBy= ZeroMatrix(mBy);
      vBy = ZeroMatrix(vBy);
      
      //Forget gate
      mWf.Init(Wf.Rows(),Wf.Cols());
      vWf.Init(Wf.Rows(),Wf.Cols());  
      mWf = ZeroMatrix(mWf);
      vWf = ZeroMatrix(vWf);
      
      mBf.Init(Bf.Rows(),Bf.Cols());
      vBf.Init(Bf.Rows(),Bf.Cols());
      mBf = ZeroMatrix(mBf);
      vBf = ZeroMatrix(vBf);
      
      //Output Gate
      mWo.Init(Wo.Rows(),Wo.Cols());
      vWo.Init(Wo.Rows(),Wo.Cols());  
      mWo = ZeroMatrix(mWo);
      vWo = ZeroMatrix(vWo);
      
      mBo.Init(Bo.Rows(),Bo.Cols());
      vBo.Init(Bo.Rows(),Bo.Cols());
      mBo = ZeroMatrix(mBo);
      vBo = ZeroMatrix(vBo);
      
      //Input Gate
      mWi.Init(Wi.Rows(),Wi.Cols());
      vWi.Init(Wi.Rows(),Wi.Cols());  
      mWi = ZeroMatrix(mWi);
      vWi = ZeroMatrix(vWi);
      
      mBi.Init(Bi.Rows(),Bi.Cols());
      vBi.Init(Bi.Rows(),Bi.Cols());
      mBi = ZeroMatrix(mBi);
      vBi = ZeroMatrix(vBi);
      
      //Candidate Gate
      mWg.Init(Wg.Rows(),Wg.Cols());
      vWg.Init(Wg.Rows(),Wg.Cols());  
      mWg = ZeroMatrix(mWg);
      vWg = ZeroMatrix(vWg);
      
      mBg.Init(Bg.Rows(),Bg.Cols());
      vBg.Init(Bg.Rows(),Bg.Cols());
      mBg = ZeroMatrix(mBg);
      vBg = ZeroMatrix(vBg);
     } 
}
matrix LSTMLayer::Output(matrix &X)
{
//Zera o estado oculto anterior

H[0].Init(N_H,1);
H[0] = ZeroMatrix(H[0]);

//Zera a memória de longo prazo passada

C[0].Init(N_H,1);
C[0] = ZeroMatrix(C[0]);

     
//Define a matrix de saída 
matrix Y; 
Y.Init(N_outs,N_ts);

//Define a saída para cada passo de tempo
matrix y;

//Define a entrada para cada passo de tempo
for(int i=0;i<N_ts;i++)
  {x[i].Init(X.Rows(),1);}


//define forget, imput, output, e canditade gate
matrix F,I,O,G; 
//==============================
//---Propagação através do tempo
for(int t=0;t<N_ts;t++)
  {
   //Preparação da entrada
   for(int i=0;i<X.Rows();i++)
     {x[t][i][0] = X[i][t];}
     
   //Concatenação
   x_h[t] = Concatenate(H[t],x[t]); 


   //Forget gate
   Zf[t] = Wf.MatMul(x_h[t])  + Bf;
   F = Sig(Zf[t]);
   //Imput gate
   Zi[t] = Wi.MatMul(x_h[t])  + Bi;
   I = Sig(Zi[t]);
   //Output gate
   Zo[t] = Wo.MatMul(x_h[t])  + Bo;
   O = Sig(Zo[t]);
   //Candidate gate
   Zg[t] = Wg.MatMul(x_h[t])  + Bg;
   G = Tanh(Zg[t]);
   //Cálculo da nova memória de longo prazo
   C[t+1] = F * C[t] + I * G;
   //Cálculo do novo estado oculto
   H[t+1] = Tanh(C[t+1]);
   H[t+1]= H[t+1] * O;
   
   //Saída
   y = Wy.MatMul(H[t+1]) + By;
   
   //Preparação da saída
   for(int i=0;i<y.Rows();i++)
     {Y[i][t] = y[i][0];}
    
  }
return Y;
}
matrix LSTMLayer::GradDescent(matrix &Ey)
{
//Gradient 
matrix Ex;
Ex.Init(N_ins,N_ts);
 
//Forget Gate
dWf = ZeroMatrix(dWf);
dBf = ZeroMatrix(dBf);

//Input Gate
dWi = ZeroMatrix(dWi);
dBi = ZeroMatrix(dBi);

//Output Gate
dWo = ZeroMatrix(dWo);
dBo = ZeroMatrix(dBo);

//Candidate Gate
dWg = ZeroMatrix(dWg);
dBg = ZeroMatrix(dBg);

//Y gate
dWy = ZeroMatrix(dWy);
dBy = ZeroMatrix(dBy);
//Erro por passo de tempo 
matrix ey;
ey.Init(Ey.Rows(),1);

//Variáveis auxiliares
matrix d_h, d_c, d_f, d_i, d_o, d_g,d_x_h;

//Estado oculto para t+1
matrix dh_next;
dh_next.Init(N_H,1);
dh_next = ZeroMatrix(dh_next);

matrix dc_next;
dc_next.Init(N_H,1);
dc_next = ZeroMatrix(dc_next);

matrix ex;
ex.Init(N_ins,1);

for(int t=(N_ts-1);t>=0;t--)
  {
   //Preparação da entrada
   for(int i=0;i<Ey.Rows();i++)
     {ey[i][0] = Ey[i][t];}
   
   //Saída
   dWy += ey.MatMul(H[t+1].Transpose());
   dBy += ey;
   
   //Hidden State Error
   d_h = Wy.Transpose();
   d_h = d_h.MatMul(ey) + dh_next;
   
   //Output gate
   d_o = d_h * Tanh(C[t+1]) * dSig(Zo[t]);
   dWo += d_o.MatMul(x_h[t].Transpose());
   dBo += d_o;
   
   //Cell state error
   d_c = d_h * dTanh(C[t+1]) * Sig(Zo[t]) + dc_next;
   
   //Forget gate
   d_f = d_c * C[t] * dSig(Zf[t]);
   dWf += d_f.MatMul(x_h[t].Transpose());
   dBf += d_f;
   
   //Input gate
   d_i = d_c * Tanh(Zg[t]) * dSig(Zi[t]);
   dWi += d_i.MatMul(x_h[t].Transpose());
   dBi += d_i;
   
   //Candidate gate
   d_g = d_c * Sig(Zi[t]) * dTanh(Zg[t]);
   dWg += d_g.MatMul(x_h[t].Transpose());
   dBg += d_g;
   
   //Concatenated Input gradient
   d_x_h = Wf.Transpose().MatMul(d_f) + Wi.Transpose().MatMul(d_i) + Wo.Transpose().MatMul(d_o) + Wg.Transpose().MatMul(d_g);
   
   //Error of Cell state of next time step
   dc_next = Sig(Zf[t]) * d_c; 
   
   //Error of Hidden State and Entry
   for(int i=0;i<d_x_h.Rows();i++)
     {if(i < N_H) dh_next[i][0] = d_x_h[i][0];
      if(i >= N_H) ex[i-N_H][0] = d_x_h[i][0];}
   
   //Escrever na matrix o erro da entrada
   for(int i=0;i<ex.Rows();i++)
     {Ex[i][t] = ex[i][0];}   
  
  }
return Ex;
}
void   LSTMLayer::Update(void)
{
if(OP == STD)
  {
   Wf = Wf - dWf*N;
   Wi = Wi - dWi*N;
   Wo = Wo - dWo*N;
   Wg = Wg - dWg*N;
   Wy = Wy - dWy*N;
   
   Bf = Bf - dBf*N;
   Bi = Bi - dBi*N;
   Bo = Bo - dBo*N;
   Bg = Bg - dBg*N;
   By = By - dBy*N;
  }

if(OP == ADAM)
  {
      it +=1;
      //saída
      mWy = AdamM(mWy,dWy,beta1);
      vWy = AdamV(vWy,dWy,beta2);
      Wy = Wy - Adam(it,mWy,vWy,beta1,beta2,alpha);
      
      mBy = AdamM(mBy,dBy,beta1);
      vBy = AdamV(vBy,dBy,beta2);
      By = By -Adam(it,mBy,vBy,beta1,beta2,alpha);
      
      //Forget
      mWf = AdamM(mWf,dWf,beta1);
      vWf = AdamV(vWf,dWf,beta2);
      Wf = Wf - Adam(it,mWf,vWf,beta1,beta2,alpha);
      
      mBf = AdamM(mBf,dBf,beta1);
      vBf = AdamV(vBf,dBf,beta2);
      Bf = Bf -Adam(it,mBf,vBf,beta1,beta2,alpha);
      
      //Output
      mWo = AdamM(mWo,dWo,beta1);
      vWo = AdamV(vWo,dWo,beta2);
      Wo = Wo - Adam(it,mWo,vWo,beta1,beta2,alpha);
      
      mBo = AdamM(mBo,dBo,beta1);
      vBo = AdamV(vBo,dBo,beta2);
      Bo = Bo -Adam(it,mBo,vBo,beta1,beta2,alpha);
      
      //Input
      mWi = AdamM(mWi,dWi,beta1);
      vWi = AdamV(vWi,dWi,beta2);
      Wi = Wi- Adam(it,mWi,vWi,beta1,beta2,alpha);
      
      mBi = AdamM(mBi,dBi,beta1);
      vBi = AdamV(vBi,dBi,beta2);
      Bi = Bi -Adam(it,mBi,vBi,beta1,beta2,alpha);
      
      //Candidate
      mWg = AdamM(mWg,dWg,beta1);
      vWg = AdamV(vWg,dWg,beta2);
      Wg = Wg - Adam(it,mWg,vWg,beta1,beta2,alpha);
      
      mBg = AdamM(mBg,dBg,beta1);
      vBg = AdamV(vBg,dBg,beta2);
      Bg = Bg -Adam(it,mBg,vBg,beta1,beta2,alpha);
  }

}
void   LSTMLayer::SaveWeights(int k,string IAname)
{
   string csv_name;
   csv_name = IAname + "\WfLayer" + IntegerToString(k);
   SaveMatrix(Wf,csv_name);
   csv_name = IAname + "\WiLayer" + IntegerToString(k);
   SaveMatrix(Wi,csv_name);
   csv_name = IAname + "\WoLayer" + IntegerToString(k);
   SaveMatrix(Wo,csv_name);
   csv_name = IAname + "\WgLayer" + IntegerToString(k);
   SaveMatrix(Wg,csv_name);
   csv_name = IAname + "\WyLayer" + IntegerToString(k);
   SaveMatrix(Wy,csv_name);
   
   csv_name = IAname + "\BfLayer" + IntegerToString(k);
   SaveMatrix(Bf,csv_name);
   csv_name = IAname + "\BiLayer" + IntegerToString(k);
   SaveMatrix(Bi,csv_name);
   csv_name = IAname + "\BoLayer" + IntegerToString(k);
   SaveMatrix(Bo,csv_name);
   csv_name = IAname + "\BgLayer" + IntegerToString(k);
   SaveMatrix(Bg,csv_name);
   csv_name = IAname + "\ByLayer" + IntegerToString(k);
   SaveMatrix(By,csv_name);
}
void   LSTMLayer::LoadWeights(int k,string IAname)
{
   string csv_name;
   csv_name = IAname + "\WfLayer" + IntegerToString(k);
   Wf = LoadMatrix(csv_name);
   csv_name = IAname + "\WiLayer" + IntegerToString(k);
   Wi = LoadMatrix(csv_name);
   csv_name = IAname + "\WoLayer" + IntegerToString(k);
   Wo = LoadMatrix(csv_name);
   csv_name = IAname + "\WgLayer" + IntegerToString(k);
   Wg = LoadMatrix(csv_name);
   csv_name = IAname + "\WyLayer" + IntegerToString(k);
   Wy = LoadMatrix(csv_name);
   
   csv_name = IAname + "\BfLayer" + IntegerToString(k);
   Bf = LoadMatrix(csv_name);
   csv_name = IAname + "\BiLayer" + IntegerToString(k);
   Bi = LoadMatrix(csv_name);
   csv_name = IAname + "\BoLayer" + IntegerToString(k);
   Bo = LoadMatrix(csv_name);
   csv_name = IAname + "\BgLayer" + IntegerToString(k);
   Bg = LoadMatrix(csv_name);
   csv_name = IAname + "\ByLayer" + IntegerToString(k);
   By = LoadMatrix(csv_name);

}
void   LSTMLayer::SetAdam(double B1, double B2, double Alph)
{

beta1 = B1;
beta2 = B2;
alpha = Alph;
}
//+------------------------------------------------------------------+
//|   biLSTM                                                         |
//+------------------------------------------------------------------+
void biLSTMLayer::InitLayer(int N_steps, int N_entries,int N_hidden, int N_outputs, double LR, Optim Op = STD)
{
//Hiperparameters 
N = LR;
OP = Op;
N_ts = N_steps;
N_ins = N_entries;
N_outs = N_outputs;
N_H = N_hidden;

//parâmetro de Xavier para inicialização. 
double xavier_;
xavier_ = N_ins + N_outs;
xavier_ = MathSqrt(6/xavier_);

//From Past to Future
//=============================

//forget gate
Wfp.Init(N_H,N_H + N_ins);
dWfp.Init(N_H,N_H + N_ins);
Wfp = InitWeights(Wfp)*xavier_;

Bfp.Init(N_H,1);
dBfp.Init(N_H,1);
Bfp = InitWeights(Bfp)*xavier_;

ArrayResize(Zfp,N_ts);
     
//input gate
Wip.Init(N_H,N_H + N_ins);
dWip.Init(N_H,N_H + N_ins);
Wip = InitWeights(Wip)*xavier_;

Bip.Init(N_H,1);
dBip.Init(N_H,1);
Bip = InitWeights(Bip)*xavier_;

ArrayResize(Zip,N_ts);

//output gate
Wop.Init(N_H,N_H + N_ins);
dWop.Init(N_H,N_H + N_ins);
Wop = InitWeights(Wop)*xavier_;

Bop.Init(N_H,1);
dBop.Init(N_H,1);
Bop = InitWeights(Bop)*xavier_;

ArrayResize(Zop,N_ts);
     
// candidate gate
Wgp.Init(N_H,N_H + N_ins);
dWgp.Init(N_H,N_H + N_ins);
Wgp = InitWeights(Wgp)*xavier_;

Bgp.Init(N_H,1);
dBgp.Init(N_H,1);
Bgp = InitWeights(Bgp)*xavier_;

ArrayResize(Zgp,N_ts);

//Output
Wyp.Init(N_outs,N_H);
dWyp.Init(N_outs,N_H);
Wyp = InitWeights(Wyp)*xavier_;


Byp.Init(N_outs,1);
dByp.Init(N_outs,1);
Byp = InitWeights(Byp)*xavier_;

ArrayResize(Zyp,N_ts);

//Estado Oculto
ArrayResize(Hp,N_ts+1);

//Memória de longo prazo
ArrayResize(Cp,N_ts+1);

//Entrada 
ArrayResize(xp,N_ts);
ArrayResize(x_hp,N_ts);

   if(OP == ADAM)
     {
      it = 0;
      beta1 = 0.9;
      beta2 = 0.999;
      alpha = N;
      //Saída
      mWyp.Init(Wyp.Rows(),Wyp.Cols());
      vWyp.Init(Wyp.Rows(),Wyp.Cols());  
      mWyp = ZeroMatrix(mWyp);
      vWyp = ZeroMatrix(vWyp);
      
      mByp.Init(Byp.Rows(),Byp.Cols());
      vByp.Init(Byp.Rows(),Byp.Cols());
      mByp= ZeroMatrix(mByp);
      vByp = ZeroMatrix(vByp);
      
      //Forget gate
      mWfp.Init(Wfp.Rows(),Wfp.Cols());
      vWfp.Init(Wfp.Rows(),Wfp.Cols());  
      mWfp = ZeroMatrix(mWfp);
      vWfp = ZeroMatrix(vWfp);
      
      mBfp.Init(Bfp.Rows(),Bfp.Cols());
      vBfp.Init(Bfp.Rows(),Bfp.Cols());
      mBfp = ZeroMatrix(mBfp);
      vBfp = ZeroMatrix(vBfp);
      
      //Output Gate
      mWop.Init(Wop.Rows(),Wop.Cols());
      vWop.Init(Wop.Rows(),Wop.Cols());  
      mWop = ZeroMatrix(mWop);
      vWop = ZeroMatrix(vWop);
      
      mBop.Init(Bop.Rows(),Bop.Cols());
      vBop.Init(Bop.Rows(),Bop.Cols());
      mBop = ZeroMatrix(mBop);
      vBop = ZeroMatrix(vBop);
      
      //Input Gate
      mWip.Init(Wip.Rows(),Wip.Cols());
      vWip.Init(Wip.Rows(),Wip.Cols());  
      mWip = ZeroMatrix(mWip);
      vWip = ZeroMatrix(vWip);
      
      mBip.Init(Bip.Rows(),Bip.Cols());
      vBip.Init(Bip.Rows(),Bip.Cols());
      mBip = ZeroMatrix(mBip);
      vBip = ZeroMatrix(vBip);
      
      //Candidate Gate
      mWgp.Init(Wgp.Rows(),Wgp.Cols());
      vWgp.Init(Wgp.Rows(),Wgp.Cols());  
      mWgp = ZeroMatrix(mWgp);
      vWgp = ZeroMatrix(vWgp);
      
      mBgp.Init(Bgp.Rows(),Bgp.Cols());
      vBgp.Init(Bgp.Rows(),Bgp.Cols());
      mBgp = ZeroMatrix(mBgp);
      vBgp = ZeroMatrix(vBgp);
     } 
//From future to past
//forget gate
Wff.Init(N_H,N_H + N_ins);
dWff.Init(N_H,N_H + N_ins);
Wff = InitWeights(Wff)*xavier_;

Bff.Init(N_H,1);
dBff.Init(N_H,1);
Bff = InitWeights(Bff)*xavier_;

ArrayResize(Zff,N_ts);
     
//input gate
Wif.Init(N_H,N_H + N_ins);
dWif.Init(N_H,N_H + N_ins);
Wif = InitWeights(Wif)*xavier_;

Bif.Init(N_H,1);
dBif.Init(N_H,1);
Bif = InitWeights(Bif)*xavier_;

ArrayResize(Zif,N_ts);
//output gate
Wof.Init(N_H,N_H + N_ins);
dWof.Init(N_H,N_H + N_ins);
Wof = InitWeights(Wof)*xavier_;

Bof.Init(N_H,1);
dBof.Init(N_H,1);
Bof = InitWeights(Bof)*xavier_;

ArrayResize(Zof,N_ts);     
// candidate gate
Wgf.Init(N_H,N_H + N_ins);
dWgf.Init(N_H,N_H + N_ins);
Wgf = InitWeights(Wgf)*xavier_;

Bgf.Init(N_H,1);
dBgf.Init(N_H,1);
Bgf = InitWeights(Bgf)*xavier_;

ArrayResize(Zgf,N_ts);
//Output
Wyf.Init(N_outs,N_H);
dWyf.Init(N_outs,N_H);
Wyf = InitWeights(Wyf)*xavier_;


Byf.Init(N_outs,1);
dByf.Init(N_outs,1);
Byf = InitWeights(Byf)*xavier_;

ArrayResize(Zyf,N_ts);

//Estado Oculto
ArrayResize(Hf,N_ts+1);

//Memória de longo prazo
ArrayResize(Cf,N_ts+1);

//Entrada 
ArrayResize(xf,N_ts);
ArrayResize(x_hf,N_ts);

   if(OP == ADAM)
     {
      it = 0;
      beta1 = 0.9;
      beta2 = 0.999;
      alpha = N;
      //Saída
      mWyf.Init(Wyf.Rows(),Wyf.Cols());
      vWyf.Init(Wyf.Rows(),Wyf.Cols());  
      mWyf = ZeroMatrix(mWyf);
      vWyf = ZeroMatrix(vWyf);
      
      mByf.Init(Byf.Rows(),Byf.Cols());
      vByf.Init(Byf.Rows(),Byf.Cols());
      mByf= ZeroMatrix(mByf);
      vByf = ZeroMatrix(vByf);
      
      //Forget gate
      mWff.Init(Wff.Rows(),Wff.Cols());
      vWff.Init(Wff.Rows(),Wff.Cols());  
      mWff = ZeroMatrix(mWff);
      vWff = ZeroMatrix(vWff);
      
      mBff.Init(Bff.Rows(),Bff.Cols());
      vBff.Init(Bff.Rows(),Bff.Cols());
      mBff = ZeroMatrix(mBff);
      vBff = ZeroMatrix(vBff);
      
      //Output Gate
      mWof.Init(Wof.Rows(),Wof.Cols());
      vWof.Init(Wof.Rows(),Wof.Cols());  
      mWof = ZeroMatrix(mWof);
      vWof = ZeroMatrix(vWof);
      
      mBof.Init(Bof.Rows(),Bof.Cols());
      vBof.Init(Bof.Rows(),Bof.Cols());
      mBof = ZeroMatrix(mBof);
      vBof = ZeroMatrix(vBof);
      
      //Input Gate
      mWif.Init(Wif.Rows(),Wif.Cols());
      vWif.Init(Wif.Rows(),Wif.Cols());  
      mWif = ZeroMatrix(mWif);
      vWif = ZeroMatrix(vWif);
      
      mBif.Init(Bif.Rows(),Bif.Cols());
      vBif.Init(Bif.Rows(),Bif.Cols());
      mBif = ZeroMatrix(mBif);
      vBif = ZeroMatrix(vBif);
      
      //Candidate Gate
      mWgf.Init(Wgf.Rows(),Wgf.Cols());
      vWgf.Init(Wgf.Rows(),Wgf.Cols());  
      mWgf = ZeroMatrix(mWgf);
      vWgf = ZeroMatrix(vWgf);
      
      mBgf.Init(Bgf.Rows(),Bgf.Cols());
      vBgf.Init(Bgf.Rows(),Bgf.Cols());
      mBgf = ZeroMatrix(mBgf);
      vBgf = ZeroMatrix(vBgf);
     } 
}


matrix biLSTMLayer::Output(matrix &X)
{
//Zera o estado oculto anterior

Hp[0].Init(N_H,1);
Hp[0] = ZeroMatrix(Hp[0]);

//Zera a memória de longo prazo passada

Cp[0].Init(N_H,1);
Cp[0] = ZeroMatrix(Cp[0]);

     
//Define a matrix de saída 
matrix Yp; 
Yp.Init(N_outs,N_ts);

//Define a saída para cada passo de tempo
matrix yp;

//Define a entrada para cada passo de tempo
for(int i=0;i<N_ts;i++)
  {xp[i].Init(X.Rows(),1);}


//define forget, imput, output, e canditade gate
matrix Fp,Ip,Op,Gp; 
//==============================
//---Propagação através do tempo
for(int t=0;t<N_ts;t++)
  {
   //Preparação da entrada
   for(int i=0;i<X.Rows();i++)
     {xp[t][i][0] = X[i][t];}
     
   //Concatenação
   x_hp[t] = Concatenate(Hp[t],xp[t]); 


   //Forget gate
   Zfp[t] = Wfp.MatMul(x_hp[t])  + Bfp;
   Fp = Sig(Zfp[t]);
   //Imput gate
   Zip[t] = Wip.MatMul(x_hp[t])  + Bip;
   Ip = Sig(Zip[t]);
   //Output gate
   Zop[t] = Wop.MatMul(x_hp[t])  + Bop;
   Op = Sig(Zop[t]);
   //Candidate gate
   Zgp[t] = Wgp.MatMul(x_hp[t])  + Bgp;
   Gp = Tanh(Zgp[t]);
   //Cálculo da nova memória de longo prazo
   Cp[t+1] = Fp * Cp[t] + Ip * Gp;
   //Cálculo do novo estado oculto
   Hp[t+1] = Tanh(Cp[t+1]);
   Hp[t+1]= Hp[t+1] * Op;
   
   //Saída
   yp = Wyp.MatMul(Hp[t+1]) + Byp;
   
   //Preparação da saída
   for(int i=0;i<yp.Rows();i++)
     {Yp[i][t] = yp[i][0];}
    
  }

//Zera o estado oculto posterior

Hf[N_ts].Init(N_H,1);
Hf[N_ts] = ZeroMatrix(Hf[N_ts]);

//Zera a memória de longo prazo passada

Cf[N_ts].Init(N_H,1);
Cf[N_ts] = ZeroMatrix(Cf[N_ts]);
  
//Define a matrix de saída 
matrix Yf; 
Yf.Init(N_outs, N_ts);

//Define a saída para cada passo de tempo
matrix yf;

//Define a entrada para cada passo de tempo
for(int i=0;i<N_ts;i++)
  {xf[i].Init(X.Rows(),1);}


//define forget, imput, output, e canditade gate
matrix Ff,If,Of,Gf; 
//==============================
//---Propagação através do tempo
for(int t=(N_ts-1);t>=0;t--)
  {
   //Preparação da entrada
   for(int i=0;i<X.Rows();i++)
     {xf[t][i][0] = X[i][t];}
     
   //Concatenação
   x_hf[t] = Concatenate(Hf[t+1],xf[t]); 


   //Forget gate
   Zff[t] = Wff.MatMul(x_hf[t])  + Bff;
   Ff = Sig(Zff[t]);
   //Imput gate
   Zif[t] = Wif.MatMul(x_hf[t])  + Bif;
   If = Sig(Zif[t]);
   //Output gate
   Zof[t] = Wof.MatMul(x_hf[t])  + Bof;
   Of = Sig(Zof[t]);
   //Candidate gate
   Zgf[t] = Wgf.MatMul(x_hf[t])  + Bgf;
   Gf = Tanh(Zgf[t]);
   //Cálculo da nova memória de longo prazo
   Cf[t] = Ff * Cf[t+1] + If * Gf;
   //Cálculo do novo estado oculto
   Hf[t] = Tanh(Cf[t]);
   Hf[t] = Hf[t] * Of;
   
   //Saída
   yf = Wyf.MatMul(Hf[t]) + Byf;
   
   //Preparação da saída
   for(int i=0;i<yf.Rows();i++)
     {Yf[i][t] = yf[i][0];}
    
  }
matrix Y;
Y = Yp + Yf;

return Y;
}

matrix biLSTMLayer::GradDescent(matrix &Ey)
{
//Gradient 
matrix Exp;
Exp.Init(N_ins,N_ts);
 
//Forget Gate
dWfp = ZeroMatrix(dWfp);
dBfp = ZeroMatrix(dBfp);

//Input Gate
dWip = ZeroMatrix(dWip);
dBip = ZeroMatrix(dBip);

//Output Gate
dWop = ZeroMatrix(dWop);
dBop = ZeroMatrix(dBop);

//Candidate Gate
dWgp = ZeroMatrix(dWgp);
dBgp = ZeroMatrix(dBgp);

//Y gate
dWyp = ZeroMatrix(dWyp);
dByp = ZeroMatrix(dByp);
//Erro por passo de tempo 
matrix eyp;
eyp.Init(Ey.Rows(),1);

//Variáveis auxiliares
matrix d_hp, d_cp, d_fp, d_ip, d_op, d_gp, d_x_hp;

//Estado oculto para t+1
matrix dh_nextp;
dh_nextp.Init(N_H,1);
dh_nextp = ZeroMatrix(dh_nextp);

matrix dc_nextp;
dc_nextp.Init(N_H,1);
dc_nextp = ZeroMatrix(dc_nextp);

matrix ex_p;
ex_p.Init(N_ins,1);

for(int t=(N_ts-1);t>=0;t--)
  {
   //Preparação da entrada
   for(int i=0;i<Ey.Rows();i++)
     {eyp[i][0] = Ey[i][t];}
   
   //Saída
   dWyp += eyp.MatMul(Hp[t+1].Transpose());
   dByp += eyp;
   
   //Hidden State Error
   d_hp = Wyp.Transpose();
   d_hp = d_hp.MatMul(eyp) + dh_nextp;
   
   //Output gate
   d_op = d_hp * Tanh(Cp[t+1]) * dSig(Zop[t]);
   dWop += d_op.MatMul(x_hp[t].Transpose());
   dBop += d_op;
   
   //Cell state error
   d_cp = d_hp * dTanh(Cp[t+1]) * Sig(Zop[t]) + dc_nextp;
   
   //Forget gate
   d_fp = d_cp * Cp[t] * dSig(Zfp[t]);
   dWfp += d_fp.MatMul(x_hp[t].Transpose());
   dBfp += d_fp;
   
   //Input gate
   d_ip = d_cp * Tanh(Zgp[t]) * dSig(Zip[t]);
   dWip += d_ip.MatMul(x_hp[t].Transpose());
   dBip += d_ip;
   
   //Candidate gate
   d_gp = d_cp * Sig(Zip[t]) * dTanh(Zgp[t]);
   dWgp += d_gp.MatMul(x_hp[t].Transpose());
   dBgp += d_gp;
   
   //Concatenated Input gradient
   d_x_hp = Wfp.Transpose().MatMul(d_fp) + Wip.Transpose().MatMul(d_ip) + Wop.Transpose().MatMul(d_op) + Wgp.Transpose().MatMul(d_gp);
   
   //Error of Cell state of next time step
   dc_nextp = Sig(Zfp[t]) * d_cp; 
   
   //Error of Hidden State and Entry
   for(int i=0;i<d_x_hp.Rows();i++)
     {if(i < N_H) dh_nextp[i][0] = d_x_hp[i][0];
      if(i >= N_H) ex_p[i-N_H][0] = d_x_hp[i][0];}
   
   //Escrever na matrix o erro da entrada
   for(int i=0;i<ex_p.Rows();i++)
     {Exp[i][t] = ex_p[i][0];}   
  
  }

//Gradient 
matrix Exf;
Exf.Init(N_ins,N_ts);
 
//Forget Gate
dWff = ZeroMatrix(dWff);
dBff = ZeroMatrix(dBff);

//Input Gate
dWif = ZeroMatrix(dWif);
dBif = ZeroMatrix(dBif);

//Output Gate
dWof = ZeroMatrix(dWof);
dBof = ZeroMatrix(dBof);

//Candidate Gate
dWgf = ZeroMatrix(dWgf);
dBgf = ZeroMatrix(dBgf);

//Y gate
dWyf = ZeroMatrix(dWyf);
dByf = ZeroMatrix(dByf);
//Erro por passo de tempo 
matrix eyf;
eyf.Init(Ey.Rows(),1);

//Variáveis auxiliares
matrix d_hf, d_cf, d_ff, d_if, d_of, d_gf,d_x_hf;

//Estado oculto para t+1
matrix dh_nextf;
dh_nextf.Init(N_H,1);
dh_nextf = ZeroMatrix(dh_nextf);

matrix dc_nextf;
dc_nextf.Init(N_H,1);
dc_nextf = ZeroMatrix(dc_nextf);

matrix ex_f;
ex_f.Init(N_ins,1);

for(int t=0;t<N_ts;t++)
  {
   //Preparação da entrada
   for(int i=0;i<Ey.Rows();i++)
     {eyf[i][0] = Ey[i][t];}
   
   //Saída
   dWyf += eyf.MatMul(Hf[t].Transpose());
   dByf += eyf;
   
   //Hidden State Error
   d_hf = Wyf.Transpose();
   d_hf = d_hf.MatMul(eyf) + dh_nextf;
   
   //Output gate
   d_of = d_hf * Tanh(Cf[t]) * dSig(Zof[t]);
   dWof += d_of.MatMul(x_hf[t].Transpose());
   dBof += d_of;
   
   //Cell state error
   d_cf = d_hf * dTanh(Cf[t]) * Sig(Zof[t]) + dc_nextf;
   
   //Forget gate
   d_ff = d_cf * Cf[t+1] * dSig(Zff[t]);
   dWff += d_ff.MatMul(x_hf[t].Transpose());
   dBff += d_ff;
   
   //Input gate
   d_if = d_cf * Tanh(Zgf[t]) * dSig(Zif[t]);
   dWif += d_if.MatMul(x_hf[t].Transpose());
   dBif += d_if;
   
   //Candidate gate
   d_gf = d_cf * Sig(Zif[t]) * dTanh(Zgf[t]);
   dWgf += d_gf.MatMul(x_hf[t].Transpose());
   dBgf += d_gf;
   
   //Concatenated Input gradient
   d_x_hf = Wff.Transpose().MatMul(d_ff) + Wif.Transpose().MatMul(d_if) + Wof.Transpose().MatMul(d_of) + Wgf.Transpose().MatMul(d_gf);
   
   //Error of Cell state of next time step
   dc_nextf = Sig(Zff[t]) * d_cf; 
   
   //Error of Hidden State and Entry
   for(int i=0;i<d_x_hf.Rows();i++)
     {if(i < N_H) dh_nextf[i][0] = d_x_hf[i][0];
      if(i >= N_H) ex_f[i-N_H][0] = d_x_hf[i][0];}
   
   //Escrever na matrix o erro da entrada
   for(int i=0;i<ex_f.Rows();i++)
     {Exf[i][t] = ex_f[i][0];}   
  
  }
matrix Ex;

Ex = Exp + Exf; 
return Ex;
}

void   biLSTMLayer::Update(void)
{
if(OP == STD)
  {
   //Past 
   Wfp = Wfp - dWfp*N;
   Wip = Wip - dWip*N;
   Wop = Wop - dWop*N;
   Wgp = Wgp - dWgp*N;
   Wyp = Wyp - dWyp*N;
   
   Bfp = Bfp - dBfp*N;
   Bip = Bip - dBip*N;
   Bop = Bop - dBop*N;
   Bgp = Bgp - dBgp*N;
   Byp = Byp - dByp*N;
   
   //Future
   Wff = Wff - dWff*N;
   Wif = Wif - dWif*N;
   Wof = Wof - dWof*N;
   Wgf = Wgf - dWgf*N;
   Wyf = Wyf - dWyf*N;
   
   Bff = Bff - dBff*N;
   Bif = Bif - dBif*N;
   Bof = Bof - dBof*N;
   Bgf = Bgf - dBgf*N;
   Byf = Byf - dByf*N;
  }
if(OP == ADAM)
  {
      it +=1;
      //saída
      mWyp = AdamM(mWyp,dWyp,beta1);
      vWyp = AdamV(vWyp,dWyp,beta2);
      Wyp = Wyp - Adam(it,mWyp,vWyp,beta1,beta2,alpha);
      
      mByp = AdamM(mByp,dByp,beta1);
      vByp = AdamV(vByp,dByp,beta2);
      Byp = Byp -Adam(it,mByp,vByp,beta1,beta2,alpha);
      
      //Forget
      mWfp = AdamM(mWfp,dWfp,beta1);
      vWfp = AdamV(vWfp,dWfp,beta2);
      Wfp = Wfp - Adam(it,mWfp,vWfp,beta1,beta2,alpha);
      
      mBfp = AdamM(mBfp,dBfp,beta1);
      vBfp = AdamV(vBfp,dBfp,beta2);
      Bfp = Bfp -Adam(it,mBfp,vBfp,beta1,beta2,alpha);
      
      //Output
      mWop = AdamM(mWop,dWop,beta1);
      vWop = AdamV(vWop,dWop,beta2);
      Wop = Wop - Adam(it,mWop,vWop,beta1,beta2,alpha);
      
      mBop = AdamM(mBop,dBop,beta1);
      vBop = AdamV(vBop,dBop,beta2);
      Bop = Bop -Adam(it,mBop,vBop,beta1,beta2,alpha);
      
      //Input
      mWip = AdamM(mWip,dWip,beta1);
      vWip = AdamV(vWip,dWip,beta2);
      Wip = Wip- Adam(it,mWip,vWip,beta1,beta2,alpha);
      
      mBip = AdamM(mBip,dBip,beta1);
      vBip = AdamV(vBip,dBip,beta2);
      Bip = Bip -Adam(it,mBip,vBip,beta1,beta2,alpha);
      
      //Candidate
      mWgp = AdamM(mWgp,dWgp,beta1);
      vWgp = AdamV(vWgp,dWgp,beta2);
      Wgp = Wgp - Adam(it,mWgp,vWgp,beta1,beta2,alpha);
      
      mBgp = AdamM(mBgp,dBgp,beta1);
      vBgp = AdamV(vBgp,dBgp,beta2);
      Bgp = Bgp -Adam(it,mBgp,vBgp,beta1,beta2,alpha);

      //saída
      mWyf = AdamM(mWyf,dWyf,beta1);
      vWyf = AdamV(vWyf,dWyf,beta2);
      Wyf = Wyf - Adam(it,mWyf,vWyf,beta1,beta2,alpha);
      
      mByf = AdamM(mByf,dByf,beta1);
      vByf = AdamV(vByf,dByf,beta2);
      Byf = Byf -Adam(it,mByf,vByf,beta1,beta2,alpha);
      
      //Forget
      mWff = AdamM(mWff,dWff,beta1);
      vWff = AdamV(vWff,dWff,beta2);
      Wff = Wff - Adam(it,mWff,vWff,beta1,beta2,alpha);
      
      mBff = AdamM(mBff,dBff,beta1);
      vBff = AdamV(vBff,dBff,beta2);
      Bff = Bff -Adam(it,mBff,vBff,beta1,beta2,alpha);
      
      //Output
      mWof = AdamM(mWof,dWof,beta1);
      vWof = AdamV(vWof,dWof,beta2);
      Wof = Wof - Adam(it,mWof,vWof,beta1,beta2,alpha);
      
      mBof = AdamM(mBof,dBof,beta1);
      vBof = AdamV(vBof,dBof,beta2);
      Bof = Bof -Adam(it,mBof,vBof,beta1,beta2,alpha);
      
      //Input
      mWif = AdamM(mWif,dWif,beta1);
      vWif = AdamV(vWif,dWif,beta2);
      Wif = Wif- Adam(it,mWif,vWif,beta1,beta2,alpha);
      
      mBif = AdamM(mBif,dBif,beta1);
      vBif = AdamV(vBif,dBif,beta2);
      Bif = Bif -Adam(it,mBif,vBif,beta1,beta2,alpha);
      
      //Candidate
      mWgf = AdamM(mWgf,dWgf,beta1);
      vWgf = AdamV(vWgf,dWgf,beta2);
      Wgf = Wgf - Adam(it,mWgf,vWgf,beta1,beta2,alpha);
      
      mBgf = AdamM(mBgf,dBgf,beta1);
      vBgf = AdamV(vBgf,dBgf,beta2);
      Bgf = Bgf -Adam(it,mBgf,vBgf,beta1,beta2,alpha);
  }
}

void   biLSTMLayer::SaveWeights(int k,string IAname)
{
   string csv_name;
   csv_name = IAname + "\WfpLayer" + IntegerToString(k);
   SaveMatrix(Wfp,csv_name);
   csv_name = IAname + "\WipLayer" + IntegerToString(k);
   SaveMatrix(Wip,csv_name);
   csv_name = IAname + "\WopLayer" + IntegerToString(k);
   SaveMatrix(Wop,csv_name);
   csv_name = IAname + "\WgpLayer" + IntegerToString(k);
   SaveMatrix(Wgp,csv_name);
   csv_name = IAname + "\WypLayer" + IntegerToString(k);
   SaveMatrix(Wyp,csv_name);
   
   csv_name = IAname + "\BfpLayer" + IntegerToString(k);
   SaveMatrix(Bfp,csv_name);
   csv_name = IAname + "\BipLayer" + IntegerToString(k);
   SaveMatrix(Bip,csv_name);
   csv_name = IAname + "\BopLayer" + IntegerToString(k);
   SaveMatrix(Bop,csv_name);
   csv_name = IAname + "\BgpLayer" + IntegerToString(k);
   SaveMatrix(Bgp,csv_name);
   csv_name = IAname + "\BypLayer" + IntegerToString(k);
   SaveMatrix(Byp,csv_name);
   
   csv_name = IAname + "\WffLayer" + IntegerToString(k);
   SaveMatrix(Wff,csv_name);
   csv_name = IAname + "\WifLayer" + IntegerToString(k);
   SaveMatrix(Wif,csv_name);
   csv_name = IAname + "\WofLayer" + IntegerToString(k);
   SaveMatrix(Wof,csv_name);
   csv_name = IAname + "\WgfLayer" + IntegerToString(k);
   SaveMatrix(Wgf,csv_name);
   csv_name = IAname + "\WyfLayer" + IntegerToString(k);
   SaveMatrix(Wyf,csv_name);
   
   csv_name = IAname + "\BffLayer" + IntegerToString(k);
   SaveMatrix(Bff,csv_name);
   csv_name = IAname + "\BifLayer" + IntegerToString(k);
   SaveMatrix(Bif,csv_name);
   csv_name = IAname + "\BofLayer" + IntegerToString(k);
   SaveMatrix(Bof,csv_name);
   csv_name = IAname + "\BgfLayer" + IntegerToString(k);
   SaveMatrix(Bgf,csv_name);
   csv_name = IAname + "\ByfLayer" + IntegerToString(k);
   SaveMatrix(Byf,csv_name);
}
void   biLSTMLayer::LoadWeights(int k,string IAname)
{
   string csv_name;
   csv_name = IAname + "\WfpLayer" + IntegerToString(k);
   Wfp = LoadMatrix(csv_name);
   csv_name = IAname + "\WipLayer" + IntegerToString(k);
   Wip = LoadMatrix(csv_name);
   csv_name = IAname + "\WopLayer" + IntegerToString(k);
   Wop = LoadMatrix(csv_name);
   csv_name = IAname + "\WgpLayer" + IntegerToString(k);
   Wgp = LoadMatrix(csv_name);
   csv_name = IAname + "\WypLayer" + IntegerToString(k);
   Wyp = LoadMatrix(csv_name);
   
   csv_name = IAname + "\BfpLayer" + IntegerToString(k);
   Bfp = LoadMatrix(csv_name);
   csv_name = IAname + "\BipLayer" + IntegerToString(k);
   Bip = LoadMatrix(csv_name);
   csv_name = IAname + "\BopLayer" + IntegerToString(k);
   Bop = LoadMatrix(csv_name);
   csv_name = IAname + "\BgpLayer" + IntegerToString(k);
   Bgp = LoadMatrix(csv_name);
   csv_name = IAname + "\BypLayer" + IntegerToString(k);
   Byp = LoadMatrix(csv_name);
   
   csv_name = IAname + "\WffLayer" + IntegerToString(k);
   Wff = LoadMatrix(csv_name);
   csv_name = IAname + "\WifLayer" + IntegerToString(k);
   Wif = LoadMatrix(csv_name);
   csv_name = IAname + "\WofLayer" + IntegerToString(k);
   Wof = LoadMatrix(csv_name);
   csv_name = IAname + "\WgfLayer" + IntegerToString(k);
   Wgf = LoadMatrix(csv_name);
   csv_name = IAname + "\WyfLayer" + IntegerToString(k);
   Wyf = LoadMatrix(csv_name);
   
   csv_name = IAname + "\BffLayer" + IntegerToString(k);
   Bff = LoadMatrix(csv_name);
   csv_name = IAname + "\BifLayer" + IntegerToString(k);
   Bif = LoadMatrix(csv_name);
   csv_name = IAname + "\BofLayer" + IntegerToString(k);
   Bof = LoadMatrix(csv_name);
   csv_name = IAname + "\BgfLayer" + IntegerToString(k);
   Bgf = LoadMatrix(csv_name);
   csv_name = IAname + "\ByfLayer" + IntegerToString(k);
   Byf = LoadMatrix(csv_name);

}
void   biLSTMLayer::SetAdam(double B1, double B2, double Alph)
{

beta1 = B1;
beta2 = B2;
alpha = Alph;
}
