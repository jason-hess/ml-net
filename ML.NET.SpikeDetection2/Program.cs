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
            var counts = new[] { 0, 1, 1, 0, 2, 1, 0, 0, 1, 1, 50, 0, 1, 0, 2, 1, 0, 1 };

            var mlContext = new MLContext();
            var estimator = mlContext.Transforms.DetectIidSpike(nameof(Output.Prediction), nameof(Input.Count), confidence: 99, pvalueHistoryLength: counts.Length / 4);
            ITransformer transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(new List<Input>()));
            var input = counts.Select(x => new Input { Count = x });
            IDataView transformedData = transformer.Transform(mlContext.Data.LoadFromEnumerable(input));
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
        public float Count { get; set; }
    }
}
