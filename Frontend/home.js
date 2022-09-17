function getFirstorLastDate() {
  var uiStock = document.getElementsByName("uiStock");
  for(var i in uiStock) {
    if(uiStock[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function onClickedEstimatePrice() {
  console.log("Estimate price button clicked");
  var firstDate = getFirstorLastDate();
  var lastDate = getLastDate();

window.onload = onPageLoad;
