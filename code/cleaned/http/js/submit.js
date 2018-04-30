$("#PostImageSubmit").click(function() {
  
  var $this = $(this);
  var loadingText = '<i class="fa fa-circle-o-notch fa-spin"></i> loading...';
  if ($(this).html() !== loadingText) {
    $this.data('original-text', $(this).html());
    $this.html(loadingText);
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
  //Replace with your server function
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

  request.done(function( msg ) {
    console.log(msg);
	document.write(msg);
  });
});

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
			});