#include<Canvas\Canvas.mqh>
CCanvas canvas;
#include <Graphics\Graphic.mqh>
CGraphic graphic;
//---

class TimeSeries
  {
public:
   //Plotar o Dataset
   void     PlotDataset(matrix &dataset,int colum);
   //Plotar teste
   void     PlotTest(matrix &y_true, matrix &y_pred);
   //ler Dataset
   matrix   ReadDataset(string M_name);
   //Escrever Dataset
   void     WriteDataset(matrix &M, string M_name);
   //Devolve a matrix com os passos de tempo para passar pela rede neural
   matrix   Features(ulong i,ulong N_steps, matrix &M);
   //Devolve a o valor real para treinar a rede neural
   matrix   RealOutput(ulong i, ulong j, ulong N_steps, matrix &M);
   //Devolve a matriz com as saídas reais
   matrix   OutputTest(int j, ulong N_steps, matrix &M_test);
   //Normaliza o dataset
   matrix   NormalizeDataset(matrix &dataset);
   //Separa um dataset de teste, segundo a porcentagem de teste
   matrix   DatasetTest(matrix &dataset, double percentage); 
   //Separa um dataset de treino, segundo a porcentagem de treino
   matrix   DatasetTrain(matrix &dataset, double percentage);
  
  
  //=============================
  //Calcula a média de uma coluna de uma matriz
  double Mean (matrix &X, ulong j);
  //Calcula o desvio padrão de uma coluna de uma matriz
  double StdDeviation(matrix &X, ulong j);
  //Calcula o valor máximo de uma coluna de uma matriz
  double Max(matrix &X,ulong j);
  //Calcula o valor mínimo de uma coluna de uma matriz
  double Min(matrix &X,ulong j);
  
  };
  
//+------------------------------------------------------------------+
//|   Methodes                                                       |
//+------------------------------------------------------------------+
matrix TimeSeries::Features(ulong i, ulong N_steps, matrix &M)
{
matrix X; 
X.Init(N_steps,M.Cols()); 

for(int r=i;r<i+N_steps;r++)
  {for(int j=0;j<M.Cols();j++)
     {X[r-i][j] = M[r][j];}}
      
X = X.Transpose();
return X;
}

matrix TimeSeries::RealOutput(ulong i, ulong j, ulong N_steps,matrix &M)
{
matrix Y;
Y.Init(1,1);
Y[0][0] = M[N_steps+i][j];
return Y; 
}

matrix TimeSeries::OutputTest(int j, ulong N_steps,matrix &M_test)
{
matrix Y_true;
Y_true.Init(M_test.Rows()-N_steps,1);

for(int i=N_steps;i<M_test.Rows();i++)
  {Y_true[i-N_steps][0] = M_test[i][j];}

return Y_true;
}

void TimeSeries::PlotDataset(matrix &dataset, int colum)
{
int N_entries,N_samples; 
N_entries = dataset.Cols();
N_samples = dataset.Rows();

//Entrada
double x[];
ArrayResize(x,N_samples); 
for(int i=0;i<N_samples;i++)
  {x[i] = i;}

//Saídas
double y[];
ArrayResize(y,N_samples);
for(int i=0;i<N_samples;i++)
  {y[i] = dataset[i][colum];}

//Plot

   //CGraphic graphic;
   
   /*bool  Create(
   const long    chart,      // chart ID
   const string  name,       // name
   const int     subwin,     // sub-window index
   const int     x1,         // x1 coordinate (canto superior esquerdo)
   const int     y1,         // y1 coordinate (canto superior esquerdo)
   const int     x2,         // x2 coordinate (canto inferior direito)
   const int     y2          // y2 coordinate (canto inferior direito)
   )*/
   // A função .Create é dada acima. 
   graphic.Create(0,"Graphic",0,30,30,1500,900);

   //Adiciona os gráficos na curva.
   CCurve *Curve1 = graphic.CurveAdd(x,y,CURVE_LINES);    // Os trechos "CCurve *Curve1 = " não
   //CCurve *Curve2 = graphic.CurveAdd(x0,y2,CURVE_LINES);    // são necessários. 
   //Muda os nomes dos gráficos
   graphic.XAxis().Name("X - axis");
   //Sem o tamanho da fonte, não funciona.       
   graphic.XAxis().NameSize(12);          
   graphic.YAxis().Name("Y - axis");      
   graphic.YAxis().NameSize(12);    
   graphic.CurvePlotAll();
   graphic.Update();
}

void     TimeSeries::PlotTest(matrix &y_true, matrix &y_pred)
{
double Y_true[], Y_pred[],x[];
ArrayResize(Y_true,y_true.Rows());
ArrayResize(Y_pred,y_pred.Rows());
ArrayResize(x,y_true.Rows());

if(ArraySize(Y_true) != ArraySize(Y_pred))
   {Print("Y_true and Y_pred with diferent sizes");}


//Escreve os vetores para serem plotados 
for(int i=0;i<ArraySize(Y_true);i++)
  {x[i] = i;  
   Y_true[i] = y_true[i][0];
   Y_pred[i] = y_pred[i][0];}
   

//Plot

   //CGraphic graphic;
   
   /*bool  Create(
   const long    chart,      // chart ID
   const string  name,       // name
   const int     subwin,     // sub-window index
   const int     x1,         // x1 coordinate (canto superior esquerdo)
   const int     y1,         // y1 coordinate (canto superior esquerdo)
   const int     x2,         // x2 coordinate (canto inferior direito)
   const int     y2          // y2 coordinate (canto inferior direito)
   )*/
   // A função .Create é dada acima. 
   graphic.Create(0,"Graphic",0,30,30,950,550);

   //Adiciona os gráficos na curva.
   CCurve *Curve1 = graphic.CurveAdd(x,Y_true,CURVE_LINES);    // Os trechos "CCurve *Curve1 = " não
   CCurve *Curve2 = graphic.CurveAdd(x,Y_pred,CURVE_LINES);    // são necessários. 
   //Muda os nomes dos gráficos
   graphic.XAxis().Name("X - axis");
   //Sem o tamanho da fonte, não funciona.       
   graphic.XAxis().NameSize(12);          
   graphic.YAxis().Name("Y - axis");      
   graphic.YAxis().NameSize(12);    
   graphic.CurvePlotAll();
   graphic.Update();   

}
matrix   TimeSeries::NormalizeDataset(matrix &dataset)
{
matrix X;
X = dataset; 
for(int j=0;j<X.Cols();j++)
  {double mean;
   double std;
   double min;
   double max; 
   mean = Mean(X,j);
   std = StdDeviation(X,j);
   
   int outlier_filter =3;
   //Tira a média
   for(int i=0;i<X.Rows();i++)
     {X[i][j] = X[i][j] - mean;}
   
   //Tira os Outliers
   for(int l=0;l<outlier_filter;l++)
     {
      
      //Tira a primeira camada de outliers
      for(int i=0;i<X.Rows();i++)
        {if(MathAbs(X[i][j]) > 3*std) X[i][j] = 0;}
        
      //Interpola os valores dos outliers
      for(int i=0;i<X.Rows();i++)
        {if(X[i][j] == 0) X[i][j] = (X[i-1][j] + X[i+1][j]) / 2;}
        
      //Recalcula o desvio padrão
      std = StdDeviation(X,j);      
    }
   //Divide pelo desvio padrão
   //for(int i=0;i<X.Rows();i++)
     //{X[i][j] = X[i][j] /std;}
   
   max = Max(X,j);
   min = Min(X,j); 
   
   for(int i=0;i<X.Rows();i++)
     {X[i][j] = (1.5 * X[i][j])/(max - min);}
   
  }
return X;
}
double TimeSeries::Mean (matrix &X, ulong j)
{
matrix M;
M = X;
double me;
me = 0; 
double N_samples; 
N_samples = M.Rows();
for(int i=0;i<M.Rows();i++)
  {me = me + M[i][j];}
   
me = me / N_samples;
return me;
}

double TimeSeries::StdDeviation(matrix &X, ulong j)
{
matrix M;
M = X;
double std;
ulong N_samples;
double me,sum;
me = Mean(X,j);
N_samples = M.Rows()-1;

sum = 0;
for(int i=0;i<M.Rows();i++)
  {sum = sum + MathPow(M[i][j]-me,2);}
  
sum = sum / N_samples;

std = MathSqrt(sum);
return std; 

}

double TimeSeries::Min(matrix &X, ulong j)
{
double min; 
min = 0; 
for(int i=0;i<X.Rows();i++)
  {if( X[i][j] < min) min = X[i][j];}

return min; 
}

double TimeSeries::Max(matrix &X, ulong j)
{
double max; 
max = 0; 
for(int i=0;i<X.Rows();i++)
  {if( X[i][j] > max) max = X[i][j];}

return max; 
}

matrix TimeSeries::ReadDataset(string M_name)
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
void TimeSeries::WriteDataset(matrix &M,string M_name)
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
matrix   TimeSeries::DatasetTest(matrix &dataset, double percentage)
{
   matrix M;
   M = dataset;
   ulong N_samples;
   N_samples = M.Rows();
   
   ulong N_samples_tr;
   N_samples_tr = N_samples*(1-percentage);
   matrix M_train,M_test;
  // M_train.Init(N_samples_tr,M.Cols());
   M_test.Init(N_samples - N_samples_tr,M.Cols());
   
   for(int i=0;i<M.Rows();i++)
     {for(int j=0;j<M.Cols();j++)
        {//if(i < N_samples_tr) M_train[i][j] = M[i][j];
         if(i >= N_samples_tr) M_test[i-N_samples_tr][j] = M[i][j];}}
         
return M_test;
} 

matrix  TimeSeries::DatasetTrain(matrix &dataset, double percentage)
{
   matrix M;
   M = dataset;
   ulong N_samples; 
   N_samples = M.Rows();
   
   ulong N_samples_tr;
   N_samples_tr = N_samples*percentage;
   matrix M_train,M_test;
   M_train.Init(N_samples_tr,M.Cols());
   //M_test.Init(N_samples - N_samples_tr,M.Cols());
   
   for(int i=0;i<M.Rows();i++)
     {for(int j=0;j<M.Cols();j++)
        {if(i < N_samples_tr) M_train[i][j] = M[i][j];}}
         //if(i >= N_samples_tr) M_test[i-N_samples_tr][j] = M[i][j];}}
return M_train;
}