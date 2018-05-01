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

$(".dropdown-menu a").click(function(){
	document.getElementById('main-nav').innerHTML = $(this).text();
});

$("#catchButton").click(function(){
	to_lang = document.getElementById('main-nav').innerHTML;
	// from_lang = 'auto';
	to_translate = document.getElementById('TextBox').getAttribute('original-answer');
	
	funcURL = "translate";
	// funcURL += "?from_lang=" + from_lang
	funcURL += "?to_lang=" + to_lang
	console.log(funcURL);
	var request = $.ajax({
		url: funcURL,
		method: "POST",
		data: to_translate,
		processData: false,
		async: true,
		cache: false,
		contentType: 'text/plain',
		timeout : 20000
	});
	request.done(function( msg ) {
		msg = msg.replace(/\r\n/g, "<br>");
		console.log(msg);
		document.getElementById('TextBox').innerHTML = msg;
	});
});