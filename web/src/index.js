function truncateString(str, num) {
  return str.length > num ? str.slice(0, num) + "..." : str;
}

document
  .getElementById("smsForm")
  .addEventListener("submit", async function (event) {
    event.preventDefault();
    const smsText = document.getElementById("smsInput").value;
    await fetch("/sms/classify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ sms: smsText }),
    })
      .then((response) => response.json())
      .then((data) => {
        document.getElementById("smsResult").innerText = `"${truncateString(
          smsText,
          20
        )}" ${data.type == "spam" ? "is spam!" : "is ham!"}`;
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });

document
  .getElementById("urlForm")
  .addEventListener("submit", async function (event) {
    event.preventDefault();
    const urlText = document.getElementById("urlInput").value;
    await fetch("/url/classify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url: urlText }),
    })
      .then((response) => response.json())
      .then((data) => {
        document.getElementById("urlResult").innerText = `"${truncateString(
          urlText,
          20
        )}" ${data.type == "spam" ? "is spam!" : "is ham!"}`;
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });
