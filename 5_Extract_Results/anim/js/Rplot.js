(function($) {
    $(document).ready(function() {
	
	$('#Rplot').scianimator({
	    'images': ['anim/images/Rplot1.png', 'anim/images/Rplot2.png'],
	    'width': 480,
	    'delay': 50,
	    'loopMode': 'none',
 'controls': ['first', 'previous',
                                     'play', 'next', 'last',
                                      'loop', 'speed'],
                                      'delayMin': 0
	});
    });
})(jQuery);
