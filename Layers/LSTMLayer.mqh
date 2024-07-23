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