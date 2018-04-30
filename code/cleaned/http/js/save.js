(function () {
	var textFile = null,
	makeTextFile = function (text) {
		var data = new Blob([text], {type: 'text/plain'});
		if (textFile !== null) {
			window.URL.revokeObjectURL(textFile);
		}
		textFile = window.URL.createObjectURL(data);
		return textFile;
	};
	textbox = document.getElementById('TextBox');
	
	var link = document.getElementById('SaveButton');
	text = textbox.innerHTML.replace(/<br>/g, "\r\n");
	link.href = makeTextFile(text);
})();