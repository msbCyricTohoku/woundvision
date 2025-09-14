document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('analysis-form');
    const startBtn = document.getElementById('start-analysis-btn');
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('progress-bar');
    const logOutput = document.getElementById('log-output');
    const scoreRange = document.getElementById('score_thresh');
    const scoreVal = document.getElementById('score-val');

    if (scoreRange && scoreVal) {
        scoreRange.addEventListener('input', function() {
            scoreVal.textContent = parseFloat(this.value).toFixed(2);
        });
    }

    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();

            startBtn.disabled = true;
            startBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
            
            logOutput.value = 'Preparing analysis...\n';
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            progressBar.classList.remove('bg-success', 'bg-danger');
            progressSection.style.display = 'block';
            
            const formData = new FormData(form);
            
            fetch('/analyze', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                if (data.job_id) {
                    startStreaming(data.job_id);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                logOutput.value += `\nCLIENT-SIDE ERROR: ${error.message}`;
                resetUI('Analysis Failed');
                progressBar.classList.add('bg-danger');
            });
        });
    }

    function startStreaming(jobId) {
        logOutput.value += `Job started with ID: ${jobId}\nConnecting to live log stream...\n-------------------------------------\n`;
        const eventSource = new EventSource(`/stream/${jobId}`);

        eventSource.addEventListener('log', function(event) {
            logOutput.value += event.data + '\n';
            logOutput.scrollTop = logOutput.scrollHeight;
        });

        eventSource.addEventListener('progress', function(event) {
            const progress = parseInt(event.data, 10);
            progressBar.style.width = progress + '%';
            progressBar.textContent = progress + '%';
        });

        eventSource.addEventListener('status', function(event) {
            if (event.data === 'completed') {
                logOutput.value += "\n-------------------------------------\nAnalysis completed successfully!";
                progressBar.style.width = '100%';
                progressBar.textContent = 'Complete';
                progressBar.classList.add('bg-success');
                window.location.href = `/results/${jobId}`;
            } else {
                logOutput.value += "\n-------------------------------------\nAnalysis failed. Check logs for details.";
                progressBar.classList.add('bg-danger');
                progressBar.textContent = 'Failed';
                resetUI('Analysis Failed');
            }
            eventSource.close();
        });

        eventSource.onerror = function() {
            logOutput.value += '\n-------------------------------------\nError: Connection to the server was lost.';
            progressBar.classList.add('bg-danger');
            progressBar.textContent = 'Connection Error';
            resetUI('Connection Lost');
            eventSource.close();
        };
    }

    function resetUI(buttonText) {
        startBtn.disabled = false;
        startBtn.innerHTML = `<i class="bi bi-play-circle-fill"></i> ${buttonText}`;
    }
});