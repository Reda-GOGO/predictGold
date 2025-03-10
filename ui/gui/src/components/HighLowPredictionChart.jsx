import React from "react";
import { Line } from "react-chartjs-2";
import { Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement, Legend, Tooltip } from "chart.js";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Legend, Tooltip);

const HighLowPredictionChart = ({ historicalData, predictedHigh, predictedLow }) => {
  // Extract dates, high, and low from historical data
  const labels = historicalData.map((data) => data.Date);
  const highData = historicalData.map((data) => data.High);
  const lowData = historicalData.map((data) => data.Low);

  // Add future days to labels for predictions
  const futureDays = predictedHigh.length;
  const futureLabels = Array.from({ length: futureDays }, (_, i) => `Day ${i + 1}`);
  const allLabels = [...labels, ...futureLabels];

  // Extend historical data with predictions
  const extendedHigh = [...highData, ...predictedHigh];
  const extendedLow = [...lowData, ...predictedLow];

  // Chart.js dataset configuration
  const data = {
    labels: allLabels,
    datasets: [
      {
        label: "Historical High",
        data: [...highData, ...Array(futureDays).fill(null)], // Fill gaps with null for predictions
        borderColor: "rgba(75, 192, 192, 0.6)",
        backgroundColor: "rgba(75, 192, 192, 0.1)",
        pointRadius: 3,
      },
      {
        label: "Historical Low",
        data: [...lowData, ...Array(futureDays).fill(null)],
        borderColor: "rgba(153, 102, 255, 0.6)",
        backgroundColor: "rgba(153, 102, 255, 0.1)",
        pointRadius: 3,
      },
      {
        label: "Predicted High",
        data: [...Array(labels.length).fill(null), ...predictedHigh],
        borderColor: "rgba(255, 99, 132, 0.6)",
        borderDash: [5, 5], // Dashed line for predictions
        pointRadius: 3,
      },
      {
        label: "Predicted Low",
        data: [...Array(labels.length).fill(null), ...predictedLow],
        borderColor: "rgba(255, 159, 64, 0.6)",
        borderDash: [5, 5], // Dashed line for predictions
        pointRadius: 3,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      tooltip: {
        mode: "index",
        intersect: false,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Date",
        },
      },
      y: {
        title: {
          display: true,
          text: "Price",
        },
      },
    },
  };

  return (
    <div className="flex w-screen h-[400px]">
      <h2>Stock Prices (Historical and Predicted)</h2>
      <Line data={data} options={options} />
    </div>
  );
};

export default HighLowPredictionChart;
