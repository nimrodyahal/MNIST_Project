$("#SaveButton").click(function(){
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
	
	text = textbox.value.replace(/<br>/g, "\r\n");
	to_lang = document.getElementById('main-nav').innerHTML;
	name = $(this).attr('download');
	name += ' - ' + to_lang + '.txt'
	
	var link = document.createElement("a");
	link.download = name;
	link.href = makeTextFile(text);
	document.body.appendChild(link);
	link.click();
	document.body.removeChild(link);
	delete link;
});

$(".dropdown-menu a").click(function(){
	document.getElementById('main-nav').innerHTML = $(this).text();
});

$("#TranslateButton").click(function(){
	to_lang = document.getElementById('main-nav').innerHTML;
	to_translate = document.getElementById('TextBox').getAttribute('original-answer');
	
	funcURL = "translate";
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
		// msg = msg.replace(/\r\n/g, "<br>");
		console.log(msg);
		textbox = document.getElementById('TextBox');
		textbox.innerHTML = msg;
		textbox.value = msg;
	});
});