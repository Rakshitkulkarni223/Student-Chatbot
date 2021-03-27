function get_name()
{
	var name=prompt("Enter your name...");
	if(name==null)
		name='';
	var div = document.createElement('div');
	div.className='chats';
	div.id='chats';

	document.getElementById('frame').appendChild(div);

	div.innerHTML = `<div class="chat_answer"></div> <div class="bg_color" id="ans" style="margin-top:15px;"><p class="text" id="answer">Hello!!${name} Welcome😊</p></div>`;
}
function speak()
{
	document.getElementById("mic").style.color="red";
	var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition;
	var recognition = new SpeechRecognition();
	recognition.onstart = function() 
	{
		setTimeout(function(){
			document.getElementById("input_text").placeholder ='Listening...';
		},2000);
		document.getElementById("input_text").placeholder="Speak Now";
	};
	recognition.onspeechend = function() 
	{
	    recognition.stop();
	}
	recognition.onresult = function(event) {
	    var transcript = event.results[0][0].transcript;
	    var confidence = event.results[0][0].confidence;
	    document.getElementById('input_text').value=transcript;
	    document.getElementById("mic").style.color="black";
	    document.getElementById("input_text").placeholder="Type a message";
	};
	recognition.start();
}