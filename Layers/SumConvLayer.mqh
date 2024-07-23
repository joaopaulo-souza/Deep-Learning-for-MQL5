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