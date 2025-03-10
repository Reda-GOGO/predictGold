import HighLowPredictionChart from "./components/HighLowPredictionChart"

function App() {

  const historicalData = [
    { Date: "2024-12-06", High: 2645.73, Low: 2613.11 },
    { Date: "2024-12-05", High: 2655.64, Low: 2623.62 },
    { Date: "2024-12-04", High: 2657.24, Low: 2632.47 },
  ];

  const predictedHigh = [2660, 2670, 2680, 2690, 2700];
  const predictedLow = [2640, 2650, 2660, 2670, 2680];
  return (
    <>
      <div className="flex w-screen h-[400px]">
        <HighLowPredictionChart
          historicalData={historicalData}
          predictedHigh={predictedHigh}
          predictedLow={predictedLow}
        />
      </div>
    </>
  )
}

export default App
