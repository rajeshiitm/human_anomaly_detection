var mediaStream = null;
var input = document.getElementById("file-input");
var videoSource = document.createElement("source");
var cam = document.getElementById("cam");
var video = document.getElementById("video");
var playBtn = document.getElementById("play");
var uploadBtn = document.getElementById("upload");

var constraints = {
  audio: false,
  video: {
    width: { ideal: 640 },
    height: { ideal: 480 },
    facingMode: "environment",
  },
};

async function getMediaStream(constraints) {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    cam.srcObject = mediaStream;
    cam.onloadedmetadata = (event) => {
      cam.play();
    };
  } catch (err) {
    console.error(err.message);
  }
}

async function switchCamera(cameraMode) {
  try {
    if (mediaStream != null && mediaStream.active) {
      var tracks = mediaStream.getVideoTracks();
      tracks.forEach((track) => {
        track.stop();
      });
      return;
    }

    cam.srcObject = null;
    constraints.video.facingMode = cameraMode;

    await getMediaStream(constraints);
  } catch (err) {
    console.error(err.message);
    alert(err.message);
  }
}

document.getElementById("switchBtn").onclick = (event) => {
  video.classList.add("hidden");
  cam.classList.remove("hidden");
  switchCamera("environment");
};

input.addEventListener("change", function () {
  const files = this.files || [];

  if (!files.length) return;

  const reader = new FileReader();

  reader.onload = function (e) {
    videoSource.setAttribute("src", e.target.result);
    video.appendChild(videoSource);
    video.load();
    video.play();
    play.addEventListener("click", () => {
      switchCamera("environment");
      video.classList.remove("hidden");
      cam.classList.add("hidden");
      videoSource.setAttribute("src", e.target.result);
      video.appendChild(videoSource);
      video.load();
      video.play();
    });
  };

  // reader.onprogress = function (e) {
  //   console.log("progress: ", Math.round((e.loaded * 100) / e.total));
  // };

  reader.readAsDataURL(files[0]);
});

uploadBtn.addEventListener("click", function () {
  input.click();
});
