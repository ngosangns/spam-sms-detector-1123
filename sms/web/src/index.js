console.log("asdasdsadd");
document
  .getElementById("smsForm")
  .addEventListener("submit", async function (event) {
    event.preventDefault();
    const smsText = document.getElementById("smsInput").value;
    await fetch("/classify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ sms: smsText }),
    })
      .then((response) => response.json())
      .then((data) => {
        document.getElementById("result").innerText =
          data.type == "spam" ? "Spam" : "Not Spam";
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });
