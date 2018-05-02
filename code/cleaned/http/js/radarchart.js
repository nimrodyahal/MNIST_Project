var ctxR = document.getElementById("radarChart").getContext('2d');
console.log("I was here");
var myRadarChart = new Chart(ctxR, {
	type: 'radar',
	data: {
		labels: ["Convoluted Filter Size", "Pooling Size", "Dropout", "Learning Rate (in hundredths)", "Epochs (in tens)", "Neuron Count (in thousands)"],
		datasets: [
			{
				label: "Neural Network 1",
				fillColor: "rgba(220,220,220,0.2)",
				strokeColor: "rgba(220,220,220,1)",
				pointColor: "rgba(220,220,220,1)",
				pointStrokeColor: "#fff",
				pointHighlightFill: "#fff",
				pointHighlightStroke: "rgba(220,220,220,1)",
				data: [5, 2, 0.5, 3, 4, 1]
			},
			{
				label: "Neural Network 2",
				fillColor: "rgba(151,187,205,0.2)",
				strokeColor: "rgba(151,187,205,1)",
				pointColor: "rgba(151,187,205,1)",
				pointStrokeColor: "#fff",
				pointHighlightFill: "#fff",
				pointHighlightStroke: "rgba(151,187,205,1)",
				data: [7, 3, 2, 3, 2, 2]
			}
		]
	},
	options: {
		responsive: true
	}    
});