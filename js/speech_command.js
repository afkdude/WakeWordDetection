let recognizer;
let modelLoaded = false;

$("#audio-switch").change(function() {
    if(this.checked){
        if(!modelLoaded){
            loadModel();
        }else{
            startListening();
        }
    }
    else {
        stopListening();
    }   
});

function loadModel(){
    $(".progress-bar").removeClass('d-none'); 
    recognizer = speechCommands.create('BROWSER_FFT');  
    Promise.all([
        recognizer.ensureModelLoaded()
      ]).then(function(){
        $(".progress-bar").addClass('d-none');
        words = recognizer.wordLabels();
        modelLoaded = true;
        startListening();
      })
}

function startListening(){
    recognizer.listen(({scores}) => {
        scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
        scores.sort((s1, s2) => s2.score - s1.score);
        if(scores[0].word === "stop"){
            $("#word-" + scores[0].word).addClass("candidate-word-active");
            setTimeout(() => {
              $("#word-" + scores[0].word).removeClass("candidate-word-active");
            }, 1000);
        }
    }, 
    {
        probabilityThreshold: 0.70
    });
}

function stopListening(){
    recognizer.stopListening();
}