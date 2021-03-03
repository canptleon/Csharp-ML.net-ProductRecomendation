using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using Microsoft.ML.Trainers;

namespace ProductRecomendation
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            var traindata = mlContext.Data.LoadFromTextFile(path: " PATH DATA", columns: new[]
            {
                new TextLoader.Column("Label", DataKind.Single, 0),
                new TextLoader.Column(name:nameof(ProductEntry.ProductID),
                dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(0) },
                keyCount: new KeyCount(262111)),
                new TextLoader.Column(name:nameof(ProductEntry.CoPurchaseProductID),
                dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(1) },
                keyCount: new KeyCount(262111))
            },
            hasHeader: true, separatorChar: '\t');

            MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();
            options.MatrixColumnIndexColumnName = nameof(ProductEntry.ProductID);
            options.MatrixRowIndexColumnName = nameof(ProductEntry.CoPurchaseProductID);
            options.LabelColumnName = "Label";
            options.LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass;
            options.Alpha = 0.01;
            options.Lambda = 0.025;
            //options.K = 100;
            options.C = 0.00001;

            var est = mlContext.Recommendation().Trainers.MatrixFactorization(options);


            ITransformer model = est.Fit(traindata);

            var predictionengine = mlContext.Model.CreatePredictionEngine<ProductEntry, Copurchase_prediction>(model);
            var prediction = predictionengine.Predict(
                                     new ProductEntry()
                                     {
                                         ProductID = 5,
                                         CoPurchaseProductID = 6
                                     });

            Console.WriteLine("\n Predicted score is " + Math.Round(prediction.Score, 1));
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        public class Copurchase_prediction
        {
            public float Score { get; set; }
        }

        public class ProductEntry
        {
            [KeyType(count: 262111)]
            public uint ProductID { get; set; }

            [KeyType(count: 262111)]
            public uint CoPurchaseProductID { get; set; }
        }

    }
}

