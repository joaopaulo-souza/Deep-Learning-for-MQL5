//+------------------------------------------------------------------+
//|   BiLSTM Layer                                                   |
//+------------------------------------------------------------------+

class BiLSTMLayer : public DeepLearning
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
//|   BiLSTM                                                         |
//+------------------------------------------------------------------+
void BiLSTMLayer::InitLayer(int N_steps, int N_entries,int N_hidden, int N_outputs, double LR, Optim Op = STD)
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


matrix BiLSTMLayer::Output(matrix &X)
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

matrix BiLSTMLayer::GradDescent(matrix &Ey)
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

void   BiLSTMLayer::Update(void)
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

void   BiLSTMLayer::SaveWeights(int k,string IAname)
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
void   BiLSTMLayer::LoadWeights(int k,string IAname)
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
void   BiLSTMLayer::SetAdam(double B1, double B2, double Alph)
{

beta1 = B1;
beta2 = B2;
alpha = Alph;
}
