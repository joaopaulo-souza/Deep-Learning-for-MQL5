#include <TimeSeriesData.mqh>
TimeSeries time_series;
#include <DeepLearning.mqh>
DeepLearning *Layer[];
int   N_layers = 5;
Loss loss;
Metrics metrics;

void OnInit()
  {
//+------------------------------------------------------------------+
//| Hiper Parameters                                                 |
//+------------------------------------------------------------------+
   //Load Dataset
   matrix M;
   M = time_series.ReadDataset("Delhi.csv");
   
   //----------------------------------------------------------------+
   //AI Name
   string IAname;
   IAname = "DelhiIA";
   //number of epochs
   ulong epoch = 1000;
   //Learning rate
   double N = 0.001;
   //Optimization methode 
   Optim OP = ADAM;
   //number of samples
   ulong N_samples = M.Rows();
   //Dropout rate
   double drp = 0.9;
   //Percentage of samples for training
   double train_ratio = 0.8;
   //Stride of Max Pooling layers
   int stride = 2;
   
   //Number of inputs
   int N_entries = M.Cols();
   //Number of time steps 
   int N_steps = 60;
   //Column to be predicted
   int feature = 0;
 
   
   //ADAM parameters
   double Beta1 = 0.8;
   double Beta2 = 0.999;
   double Alpha = N; 
        
   //+------------------------------------------------------------------+
   //|    Neural Network creation                                     |
   //+------------------------------------------------------------------+
   
   ArrayResize(Layer,N_layers);
   
   //Architecture creation
   biLSTMLayer            *biLSTM1 = new biLSTMLayer();
   ActivationLayer      *ACT2  = new ActivationLayer();
   FlattenLayer         *FLAT3 = new FlattenLayer();
   DenseLayer           *ANN4  = new DenseLayer();
   ActivationLayer      *ACT5  = new ActivationLayer();
   
   //Layers Initialization
   biLSTM1.InitLayer(N_steps,N_entries,4,1,N,OP);
   ACT2.InitLayer(TANH);  
   ANN4.InitLayer(N_steps*1,1,N,OP);
   ACT5.InitLayer(TANH);
   
   
   //Neural Network 
   Layer[0] = biLSTM1;
   Layer[1] = ACT2;
   Layer[2] = FLAT3;
   Layer[3] = ANN4;
   Layer[4] = ACT5;
   
   //+------------------------------------------------------------------+
   //|  Otimizers parameters                                 |
   //+------------------------------------------------------------------+
   
   //ADAM
   Adam(Beta1,Beta2,Alpha); 

   
   //+------------------------------------------------------------------+
   //|    Dataset pre-processing                                        |
   //+------------------------------------------------------------------+
      
   //Shuffle  Dataset
   //M = dataset.Shuffle(M);
   
   //Normalize dataset
   M = time_series.NormalizeDataset(M);
   
   //training Dataset 
   matrix M_train;
   M_train = time_series.DatasetTrain(M,train_ratio);
   
   // testing Dataset 
   matrix M_test;
   M_test = time_series.DatasetTest(M,1-train_ratio);
   
   //+------------------------------------------------------------------+
   //|    Weights                                                        |
   //+------------------------------------------------------------------+
   
   //Load Neural Network weights
   //Load(IAname);
   
   // Training:
   Train(epoch,feature,N_steps,N,M_train);
     
   //Save the weigths
   Save(IAname);
    
    
   //+------------------------------------------------------------------+
   //|   Evaluation                                                     |
   //+------------------------------------------------------------------+
   
   
   //Output
   matrix Out;
   Out = Test(M_test, N_steps);

 
   //Compare
   matrix Y_true = time_series.OutputTest(feature,N_steps,M_test);
   
   //Plotar outputs
   time_series.PlotTest(Y_true,Out); 
   //time_series.PlotDataset(M,0);
   
   
   //Accuracy
   double mape;
   mape = metrics.MAPE(Y_true,Out);
   //int percent_accuracy;
   //accuracy = metrics.Accuracy(R,Out);
   //percent_accuracy = 100*accuracy;
   
   Print("The Mean Absolute Percentage Error of the model is ",mape);
   
   
  }
//+------------------------------------------------------------------+
//|   Functions and methods                                             |
//+------------------------------------------------------------------+

void Train(ulong epoch,int feature, ulong N_steps,double N,matrix &M)
{
   ulong N_samples;
   N_samples = M.Rows();
   
   //estimate time
   datetime date1, date2; 
   int interval, min, sec; 
   
   //Estimate training progress
   double count1,count2;
   int count;
   
   double Error; 
   //Start of training
   for(int e=0;e<epoch;e++)
     {
      Error = 0;
      for(int s=0;s<N_samples-N_steps;s++)
        {
         
         //Input and output preparation
         matrix Y_pred;
         matrix Grad;
         matrix Y_true;
         
         Y_true = time_series.RealOutput(s,feature,N_steps,M);
         Y_pred = time_series.Features(s,N_steps,M);
         
         //Forward Propagation
         for(int i=0;i<ArraySize(Layer);i++)
           { Y_pred = Layer[i].Output(Y_pred); }
           
         //Error
         Error += loss.MeanSquaredError(Y_true,Y_pred);
         Grad = loss.Grad_MSE(Y_true,Y_pred);
         
         //Backpropagation
         for(int i=ArraySize(Layer)-1;i>-1;i--)
           { Grad = Layer[i].GradDescent(Grad); }
           
         //Update Weights
         for(int i=0;i<ArraySize(Layer);i++)
           {Layer[i].Update();}  
         }
    
     //Progress bar
     count = epoch*0.01;
     if((e % (count)) == 0)
       {
         count1 = epoch;
         count2 = e;
         count2 = count2/count1;
         count = 100*count2;
         Print("training progress in ",count,"%");
         Print("Current Error = ",Error);
         date1 = TimeLocal(); 
         if(count > 0)
           {
            //Calculate time passed 
            //entre x% e x+1%
            interval = date1 - date2;
            //Estimate seconds left
            interval = interval *(100 - count);
            //Calculate minutes
            min = interval / 60;
            //Calculate seconds
            sec = interval % 60;
            Print("Remaining Time: ",min,"min ",sec,"sec");
            
           }
         date2 = TimeLocal();
       }
//+++++++++++++
     }
}

matrix Test(matrix &M_test,int N_steps)
{
   ulong N_samples;
   N_samples = M_test.Rows();
   
   matrix Y_pred;
   
   matrix Out;
   Out.Init(N_samples-N_steps,1);
   
   for(int s=0;s<N_samples-N_steps;s++)
     {   
         
         Y_pred = time_series.Features(s,N_steps,M_test);
         for(int i=0;i<ArraySize(Layer);i++)
           {//Linha responsável por anular o dropset
            Layer[i].SetDrop(1.0);
            //Saída efetivamente
            Y_pred = Layer[i].Output(Y_pred); }
         
         Out[s][0] = Y_pred[0][0];
         //   
      }
return Out;
}

void Load(string IA_name)
{
   for(int i=0;i<N_layers;i++)
     {Layer[i].LoadWeights(i,IA_name);}
}
void Save(string IA_name)
{
   for(int i=0;i<N_layers;i++)
     {Layer[i].SaveWeights(i,IA_name);}
}

void Adam(double beta1, double beta2, double alpha)
{
   for(int i=0;i<N_layers;i++)
     {Layer[i].SetAdam(beta1,beta2,alpha);}
}