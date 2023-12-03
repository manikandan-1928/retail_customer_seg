function handleSubmit(event) {
    event.preventDefault();

    var file = document.getElementById('file-upload').files[0];
    var formdata = new FormData();
    formdata.append('file', file);

    var spinner = document.getElementById('spinner');
    spinner.style.display = 'block';

    var xhr = new XMLHttpRequest();
    xhr.open('post', '/predict');
    xhr.send(formdata);

    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            document.getElementById('results').innerHTML = '';

            var response = JSON.parse(xhr.responseText);

            for (var key in response) {
                if (response.hasOwnProperty(key)) {
                    var img = document.createElement('img');
                    img.src = response[key];
                    img.style.width = '30%';
                    document.getElementById('results').appendChild(img);
                }
            }

            // Display the output saved message if available
            var output_saved_message = response['output_saved_message'];
            if (output_saved_message) {
                var p = document.createElement('p');
                p.innerHTML = output_saved_message;
                document.getElementById('results').appendChild(p);
            }

            spinner.style.display = 'none';
        }
    };
}
