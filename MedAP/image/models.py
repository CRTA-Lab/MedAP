from django.db import models
from django.contrib.auth.models import User

class Image(models.Model):
    '''
    An image that needs to be segmented/annotated.
    '''

    image = models.ImageField(upload_to='uploads/')
    mask = models.ImageField(null=True, blank=True)
    is_segmented = models.BooleanField()
    segmented_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)

    def __str__(self):
        # 'uploads/' is unnecessary
        return str(self.image.name.split('/')[-1])
