var number_of_image_slides=0;
Slider();
function Slider()
{
    var slides = document.getElementsByClassName("slideshow");
    for(var i=0;i<slides.length;i++)
    {
      slides[i].style.display = "none";
    }
    number_of_image_slides++;
    if (number_of_image_slides > slides.length)
    {
      number_of_image_slides = 1
    }
    slides[number_of_image_slides-1].style.display = "block";
    setTimeout('Slider()',2000);
}