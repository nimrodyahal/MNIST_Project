
$(document).ready( function() {
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
		document.getElementById('main-nav').innerHTML = '<font color="black">' + $(this).text() + '</font>';
	});
	
	$("#TranslateButton").click(function(){
		to_lang_raw = document.getElementById('main-nav').innerHTML;
		re = /<font color="\w*">(\w*)<\/font>/g;
		to_lang = re.exec(to_lang_raw)[1];

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
	
	var width = document.getElementById('img-upload').width;
	console.log(width);
	var width = document.getElementById('img-upload').offsetWidth;
	console.log(width);
	var width = document.getElementById('img-upload').naturalWidth;
	console.log(width);
	var width = document.getElementById('img-upload').clientWidth;
	console.log(width);
	TextArea = document.getElementById('TextBox');
	TextArea.setAttribute('style', TextArea.getAttribute('style')+' max-width:' + width + 'px;');
});