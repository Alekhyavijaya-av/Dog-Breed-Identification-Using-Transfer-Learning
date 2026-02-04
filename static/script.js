// Check if file is selected before submitting
function validateForm() {
    let fileInput = document.getElementById("imageUpload");

    if (fileInput.value === "") {
        alert("Please select an image file first!");
        return false;
    }
    return true;
}

// Show file name after selection
document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("imageUpload");
    const fileNameDisplay = document.getElementById("fileName");

    if (fileInput) {
        fileInput.addEventListener("change", function () {
            fileNameDisplay.innerText = "Selected File: " + fileInput.files[0].name;
        });
    }
});
