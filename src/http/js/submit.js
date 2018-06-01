$(document).ready( function() {
	$(document).on('change', '.btn-file :file', function() {
		var input = $(this),
		label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
		input.trigger('fileselect', [label]);
	});

	$('.btn-file :file').on('fileselect', function(event, label) {
		var input = $(this).parents('.input-group').find(':text'),
		log = label;

		if( input.length ) {
			input.val(log);
		} else {
			if( log ) alert(log);
		}
	});
	
	function readURL(input) {
		if (input.files && input.files[0]) {
			var reader = new FileReader();

			reader.onload = function (e) {
				$('#img-upload').attr('src', e.target.result);
			}

			reader.readAsDataURL(input.files[0]);
		}
	}

	$("#PostImage").change(function(){
		readURL(this);
	}); 	

	$("#PostImageSubmit").click(function() {
		var $this = $(this);
		var loadingText = '<i class="fa fa-circle-o-notch fa-spin"></i> loading...';
		
		if ($(this).html() !== loadingText) {
			$this.html(loadingText);
		} else {
			return;
		}
		
		file = $("#PostImage").get(0).files[0];
		filename = $("#PostImage").val();
		console.log(file);
		
		if (file.length === 0) {
			file = "";
		} 
		if (filename.length === 0) {
			filename = "";
		} else {
			filename = filename.substr(filename.lastIndexOf("\\") + 1);
		}
		
		funcURL = "upload";
		funcURL += "?file-name=" + filename
		console.log(funcURL);
		var request = $.ajax({
			url: funcURL,
			method: "POST",
			data: file,
			processData: false,
			async: true,
			cache: false,
			contentType: 'text/plain',
			timeout : 20000
		});
		console.log(request);
		request.done(function( msg ) {
			console.log(msg);
			if (msg == 'Success')
			{
				var link = document.createElement("a");
				link.href = 'results';
				document.body.appendChild(link);
				link.click();
				document.body.removeChild(link);
				delete link;
			}
			// document.write(msg);
		});
	});
});