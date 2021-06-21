using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;

namespace ML.NET.SalesAnomalies
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "phone-calls.csv");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(); 
            IDataView dataView = mlContext.Data.LoadFromTextFile<PhoneCallsData>(path: _dataPath, hasHeader: true, separatorChar: ',');

            int period = DetectPeriod(mlContext, dataView);
            DetectAnomaly(mlContext, dataView, period);
        }

        static int DetectPeriod(MLContext mlContext, IDataView phoneCalls)
        {
            int period = mlContext.AnomalyDetection.DetectSeasonality(phoneCalls, nameof(PhoneCallsData.value));
            Console.WriteLine("Period of the series is: {0}.", period);
            return period;
        }

        static void DetectAnomaly(MLContext mlContext, IDataView phoneCalls, int period)
        {
            var options = new SrCnnEntireAnomalyDetectorOptions()
            {
                Threshold = 0.3,
                Sensitivity = 64.0,
                DetectMode = SrCnnDetectMode.AnomalyAndMargin,
                Period = period
            };
            var outputDataView = mlContext.AnomalyDetection.DetectEntireAnomalyBySrCnn(phoneCalls, nameof(PhoneCallsPrediction.Prediction), nameof(PhoneCallsData.value), options);
            var predictions = mlContext.Data.CreateEnumerable<PhoneCallsPrediction>(
                outputDataView, reuseRowObject: false);
            Console.WriteLine("Index\tData\tAnomaly\tAnomalyScore\tMag\tExpectedValue\tBoundaryUnit\tUpperBoundary\tLowerBoundary");

            var index = 0;

            var x = mlContext.Data.CreateEnumerable<PhoneCallsData>(phoneCalls, reuseRowObject:false).ToArray();

            foreach (var p in predictions)
            {
                if (p.Prediction[0] == 1)
                {
                    Console.WriteLine(x[index].value);
                    Console.WriteLine("{0},{1},{2},{3},{4}  <-- alert is on, detecte anomaly", index,
                        p.Prediction[0], p.Prediction[3], p.Prediction[5], p.Prediction[6]);
                }
                else
                {
                    Console.WriteLine(x[index].value);
                    Console.WriteLine("{0},{1},{2},{3},{4}", index,
                        p.Prediction[0], p.Prediction[3], p.Prediction[5], p.Prediction[6]);
                }
                ++index;

            }

            Console.WriteLine("");
        }
    }

    public class PhoneCallsData
    {
        [LoadColumn(0)]
        public string timestamp;

        [LoadColumn(1)]
        public double value;
    }

    public class PhoneCallsPrediction
    {
        //vector to hold anomaly detection results. Including isAnomaly, anomalyScore, magnitude, expectedValue, boundaryUnits, upperBoundary and lowerBoundary.
        [VectorType(7)]
        public double[] Prediction { get; set; }
    }
}
