using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ML.NET.SpikeDetection2
{
    class Program
    {

        static void Main(string[] args)
        {
            //var counts = new[] { 0, 1, 1, 0, 2, 1, 0, 0, 1, 1, 0, 1, 1, 0, 2, 1, 0, 0, 1, 1, 50, 0, 1, 0, 2, 1, 0, 1 };
            //counts = new[] { 0, 1, 1, 0, 2, 1, 0, 0, 1, 1, 0, 1, 1, 3, 0, 4, 5, 5, 4, 3, 3, 0, 13, 8, 1,21, 40, 7, 7, 5, 6, 8, 33, 11, 5,
            //    2, 10, 11, 18, 14, 23, 8, 17, 15, 13, 24, 29, 15, 20, 29, 19, 18, 17, 23, 47, 7, 14, 26, 28,
            //    5, 22, 47, 22, 20, 9, 40, 6, 8, 4, 10, 10, 1, 4, 27, 3, 3, 7,  61, 6, 12, 8, 3, 1, 2, 0, 0, 2, 0,
            //    2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2 };

            var mlContext = new MLContext();
            var estimator = mlContext.Transforms.DetectIidSpike(nameof(Output.Prediction), nameof(Input.Count), confidence: 99, pvalueHistoryLength: 10);
            //ITransformer transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(new List<Input>()));
            var transformer =
                estimator.Fit(
                    mlContext.Data.LoadFromTextFile<Input>(@"C:\Users\aujasonh\Downloads\export-8d935525863cb89.csv"));
            //var input = counts.Select(x => new Input { Count = x });
            IDataView transformedData =
                transformer.Transform(
                    mlContext.Data.LoadFromTextFile<Input>(@"C:\Users\aujasonh\Downloads\export-8d935525863cb89.csv"));
            var predictions = mlContext.Data.CreateEnumerable<Output>(transformedData, false);

            foreach (var p in predictions)
            {
                Console.WriteLine($"{p.Prediction[0]}\t{p.Prediction[1]}\t{p.Prediction[2]}");
            }
        }
    }

    class Output
    {
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }

    class Input
    {
        [LoadColumn(0)]
        public float Count { get; set; }
    }
}
